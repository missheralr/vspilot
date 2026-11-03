"""Read the GPS log embedded in a 70mai dashcam MP4. Pure Python.

No ffmpeg or ffprobe. A Windows machine usually has neither, and the GPS log
is what unlocks ego speed, time-to-collision and the planner, so it must not
depend on a tool the user has to install first.

The file is parsed directly: walk the MP4 box tree, find the data track, use
its sample tables (``stsz`` + ``stsc`` + ``stco``/``co64``) to locate every
sample, and read them straight out of ``mdat``.

Verified byte-for-byte against an ffmpeg extraction of the same stream.

GPS record layout, 32 bytes, one per second::

    0-7     float64 LE   latitude,  NMEA ddmm.mmmm
    8-15    float64 LE   longitude, NMEA dddmm.mmmm
    16-19                always zero
    20-21   uint16 LE    speed, km/h
    22-23                constant 0x067e
    24      uint8        day
    25      uint8        hour (UTC)
    26      uint8        minute
    27      uint8        second
    28-29   uint16 LE    believed to be altitude in units of 0.1 m (unconfirmed)
    30      uint8        always 1 — most likely a "valid fix" flag
    31      uint8        always 0

Records carry no month or year; take those from the filename if needed.
"""

from __future__ import annotations

import math
import os
import struct

RECORD_SIZE = 32


# --- MP4 box tree ----------------------------------------------------------
def _iter_boxes(f, end: int):
    """Yield ``(name, payload_start, box_end)`` for boxes at this level."""
    while f.tell() < end - 8:
        start = f.tell()
        header = f.read(8)
        if len(header) < 8:
            return
        size, name = struct.unpack(">I4s", header)
        name = name.decode("latin-1")
        if size == 1:  # 64-bit size
            size = struct.unpack(">Q", f.read(8))[0]
            payload = start + 16
        elif size == 0:  # extends to end of file
            size = end - start
            payload = start + 8
        else:
            payload = start + 8
        if size < 8:
            return
        yield name, payload, min(start + size, end)
        f.seek(start + size)


def _find(f, start: int, end: int, path: list[str]):
    """Find boxes by path, e.g. ``["moov", "trak"]``. Returns ``[(start, end)]``."""
    found = []
    f.seek(start)
    for name, payload, box_end in _iter_boxes(f, end):
        if name == path[0]:
            if len(path) == 1:
                found.append((payload, box_end))
            else:
                found += _find(f, payload, box_end, path[1:])
    return found


def _sample_table(f, start: int, end: int):
    """Read ``stsz``/``stsc``/``stco``|``co64`` into ``[(offset, size)]``."""
    sizes: list[int] = []
    chunk_offsets: list[int] = []
    stsc: list[tuple[int, int, int]] = []

    f.seek(start)
    for name, payload, _box_end in _iter_boxes(f, end):
        f.seek(payload)
        if name == "stsz":
            _, uniform_size, count = struct.unpack(">III", f.read(12))
            if uniform_size == 0:
                sizes = list(struct.unpack(f">{count}I", f.read(4 * count)))
            else:
                sizes = [uniform_size] * count
        elif name == "stco":
            _, count = struct.unpack(">II", f.read(8))
            chunk_offsets = list(struct.unpack(f">{count}I", f.read(4 * count)))
        elif name == "co64":
            _, count = struct.unpack(">II", f.read(8))
            chunk_offsets = list(struct.unpack(f">{count}Q", f.read(8 * count)))
        elif name == "stsc":
            _, count = struct.unpack(">II", f.read(8))
            stsc = [struct.unpack(">III", f.read(12)) for _ in range(count)]

    if not (sizes and chunk_offsets and stsc):
        return []

    samples, index = [], 0
    for entry, (first_chunk, samples_per_chunk, _desc) in enumerate(stsc):
        last_chunk = (
            stsc[entry + 1][0] - 1 if entry + 1 < len(stsc) else len(chunk_offsets)
        )
        for chunk in range(first_chunk - 1, last_chunk):
            if chunk >= len(chunk_offsets):
                break
            offset = chunk_offsets[chunk]
            for _ in range(samples_per_chunk):
                if index >= len(sizes):
                    break
                samples.append((offset, sizes[index]))
                offset += sizes[index]
                index += 1
    return samples


def _handler_type(f, start: int, end: int) -> str:
    for hdlr_start, hdlr_end in _find(f, start, end, ["mdia", "hdlr"]):
        f.seek(hdlr_start + 8)
        return f.read(4).decode("latin-1", "replace")
    return "?"


# --- decoding --------------------------------------------------------------
def _nmea_to_degrees(value: float) -> float:
    degrees = int(value // 100)
    return degrees + (value - degrees * 100) / 60.0


def _looks_like_gps(record: bytes) -> bool:
    if len(record) < RECORD_SIZE or len(record) % RECORD_SIZE:
        return False
    lat, lon = struct.unpack("<dd", record[:16])
    if not (0 < lat < 9000 and 0 < lon < 18000):
        return False
    # Coordinates alone can match by chance, so check the clock and speed too.
    speed = struct.unpack("<H", record[20:22])[0]
    hour, minute, second = record[25], record[26], record[27]
    return speed <= 400 and hour <= 23 and minute <= 59 and second <= 59


def read_gps_log(path: str) -> list[dict] | None:
    """Return the GPS records, or ``None`` if this file has no log."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        for trak_start, trak_end in _find(f, 0, file_size, ["moov", "trak"]):
            handler = _handler_type(f, trak_start, trak_end)
            stbl = _find(f, trak_start, trak_end, ["mdia", "minf", "stbl"])
            if not stbl:
                continue
            samples = _sample_table(f, *stbl[0])
            if not samples:
                continue
            if not all(size == RECORD_SIZE for _offset, size in samples):
                continue
            f.seek(samples[0][0])
            if not _looks_like_gps(f.read(RECORD_SIZE)):
                continue

            records = []
            for second, (offset, size) in enumerate(samples):
                f.seek(offset)
                raw = f.read(size)
                if len(raw) < RECORD_SIZE:
                    break
                lat, lon = struct.unpack("<dd", raw[:16])
                if not (0 < lat < 9000 and 0 < lon < 18000):
                    continue
                records.append(
                    {
                        "second": second,
                        "latitude": _nmea_to_degrees(lat),
                        "longitude": _nmea_to_degrees(lon),
                        "speed_kmh": struct.unpack("<H", raw[20:22])[0],
                        "altitude_m": struct.unpack("<H", raw[28:30])[0] / 10.0,
                        "track_handler": handler,
                    }
                )
            if records:
                return records
    return None


def cross_check(records: list[dict]) -> dict:
    """Compare distance from coordinates against distance from speed.

    A large disagreement means the decode is wrong somewhere.
    """
    earth_radius = 6371000.0
    from_coords = 0.0
    for a, b in zip(records, records[1:]):
        lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
        dlat = lat2 - lat1
        dlon = math.radians(b["longitude"] - a["longitude"])
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        from_coords += 2 * earth_radius * math.asin(min(1.0, math.sqrt(h)))
    # 1 Hz, so each record covers one second
    from_speed = sum(r["speed_kmh"] for r in records) / 3.6
    return {
        "distance_from_coords_m": from_coords,
        "distance_from_speed_m": from_speed,
        "relative_error": abs(from_coords - from_speed) / max(from_speed, 1e-6),
    }


def speed_per_frame(records: list[dict], frame_count: int, fps: float):
    """Interpolate the 1 Hz log onto every frame. Returns m/s."""
    import numpy as np

    seconds = np.array([r["second"] for r in records], float)
    speed_ms = np.array([r["speed_kmh"] for r in records], float) / 3.6
    frame_times = np.arange(frame_count) / fps
    return np.interp(frame_times, seconds, speed_ms)


if __name__ == "__main__":
    import sys

    for video in sys.argv[1:]:
        rows = read_gps_log(video)
        name = os.path.basename(video)
        if not rows:
            print(f"{name}: no GPS log found")
            continue
        check = cross_check(rows)
        speeds = [r["speed_kmh"] for r in rows]
        print(
            f"{name}: {len(rows)} records, track '{rows[0]['track_handler']}', "
            f"speed {min(speeds)}-{max(speeds)} km/h"
        )
        print(
            f"   cross-check: {check['distance_from_coords_m']:.0f} m from coordinates "
            f"vs {check['distance_from_speed_m']:.0f} m from speed, "
            f"{100 * check['relative_error']:.1f}% apart"
        )
