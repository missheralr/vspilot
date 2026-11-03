#!/usr/bin/env python3
"""Run the Vision Pilot models over a dashcam video.

    python main.py clip.mp4 --out results --hz 10 --camera-height 1.32

Writes ``results/annotated.mp4`` and ``results/metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import cv2
import numpy as np

import gps_log
from calibration import DEFAULT_FOE, NET_HEIGHT, NET_WIDTH, VirtualCamera
from geometry import fit_path, objects_in_corridor, waypoints_to_world
from models import ModelBundle, find_weights
from render import PANEL_WIDTH, render_frame
from tracking import LongitudinalPlanner, SceneTracker

CSV_COLUMNS = [
    "frame",
    "time_s",
    "ego_speed_kmh",
    "lead_present",
    "lead_distance_m",
    "lead_distance_stddev_m",
    "lead_closing_speed_kmh",
    "time_to_collision_s",
    "target_acceleration_ms2",
    "lead_confidence",
    "lead_distance_raw_m",
    "objects_in_frame",
    "objects_in_corridor",
    "nearest_in_corridor_m",
    "path_curvature_1pm",
    "path_radius_m",
    "cross_track_error_m",
    "heading_error_deg",
    "look_ahead_m",
    "waypoints",
    "inference_ms",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--hz", type=float, default=10.0,
                   help="inference rate; Vision Pilot is designed for 10")
    p.add_argument("--from-s", type=float, default=0.0, dest="from_s")
    p.add_argument("--to-s", type=float, default=1e9, dest="to_s")
    p.add_argument("--camera-height", type=float, default=1.30,
                   help="camera height above the road in metres")
    p.add_argument("--foe", default=None, help="focus of expansion as 'x,y'")
    p.add_argument("--weights", default=None, help="directory holding the .onnx files")
    p.add_argument("--ego-speed-kmh", type=float, default=None,
                   help="force a constant ego speed instead of reading the GPS log")
    p.add_argument("--speed-limit-kmh", type=float, default=50.0,
                   help="speed limit used by the IDM planner")
    p.add_argument("--headway", type=float, default=1.5,
                   help="IDM desired time headway in seconds. The 1.5 default comes "
                        "from the C++. Vietnamese traffic follows far closer: 1.09 s "
                        "measured on a real clip, so 1.0 is more realistic there.")
    p.add_argument("--min-gap", type=float, default=2.0,
                   help="IDM minimum standstill gap in metres")
    p.add_argument("--no-tracking", action="store_true",
                   help="disable temporal tracking; every frame independent")
    return p


def load_ego_speed(args, video_path, frame_count, fps):
    """Per-frame ego speed in m/s, or ``None`` if unavailable."""
    if args.ego_speed_kmh is not None:
        print(f"  ego speed: forced to {args.ego_speed_kmh:.0f} km/h")
        return np.full(frame_count, args.ego_speed_kmh / 3.6)
    try:
        records = gps_log.read_gps_log(video_path)
    except Exception as exc:
        print(f"  (could not read the GPS log: {exc})")
        return None
    if not records:
        print("  ego speed: UNAVAILABLE (no GPS log in this video). "
              "Time-to-collision and acceleration need it; use --ego-speed-kmh.")
        return None
    check = gps_log.cross_check(records)
    speeds = [r["speed_kmh"] for r in records]
    print(f"  ego speed: embedded GPS log, {len(records)} records, "
          f"{min(speeds)}-{max(speeds)} km/h, distance cross-check "
          f"{100 * check['relative_error']:.1f}% apart")
    return gps_log.speed_per_frame(records, frame_count, fps)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {width}x{height} {fps:.2f} fps {frame_count} frames "
          f"({frame_count / fps:.1f} s)")

    foe = tuple(float(v) for v in args.foe.split(",")) if args.foe else DEFAULT_FOE
    camera = VirtualCamera(width, height, foe, camera_height_m=args.camera_height)
    print(camera.describe())

    models = ModelBundle(find_weights(args.weights))
    ego_speed = load_ego_speed(args, args.video, frame_count, fps)

    tracker = None if args.no_tracking else SceneTracker()
    planner = LongitudinalPlanner(
        args.speed_limit_kmh / 3.6, headway_s=args.headway, min_gap_m=args.min_gap
    )
    print(f"  tracking: {'OFF' if args.no_tracking else 'on'}   "
          f"IDM: limit {args.speed_limit_kmh:.0f} km/h, headway {args.headway:.2f} s, "
          f"min gap {args.min_gap:.1f} m")

    stride = max(1, int(round(fps / args.hz)))
    first = int(round(args.from_s * fps))
    last = min(frame_count - 1, int(round(args.to_s * fps)))
    planned = (last - first) // stride + 1
    print(f"running {args.from_s:.0f}s -> {min(args.to_s, frame_count / fps):.0f}s, "
          f"every {stride} frames (~{fps / stride:.1f} Hz) = {planned} inferences\n",
          flush=True)

    writer = cv2.VideoWriter(
        os.path.join(args.out, "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.hz,
        (NET_WIDTH + PANEL_WIDTH, NET_HEIGHT),
    )

    rows, previous_warped, done, started = [], None, 0, time.time()
    for index in range(first, last + 1, stride):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        warped, resized = camera.prepare(frame)
        current_warped = VirtualCamera.chw_imagenet(warped)
        if previous_warped is None:  # inference.cpp:150 skips the first frame
            previous_warped = current_warped
            continue

        t0 = time.time()
        out = models.run(previous_warped, current_warped, VirtualCamera.chw_01(resized))
        inference_s = time.time() - t0
        previous_warped = current_warped

        points = waypoints_to_world(out["waypoint_x"], out["waypoint_conf"], camera.H)
        path = fit_path(points)
        corridor_count, nearest = objects_in_corridor(out["detections"], camera.H)
        raw_curvature = path.curvature if path else None
        detector_lead = out["lead_confidence"] >= 0.5
        raw_distance = out["lead_distance_m"] if detector_lead else None
        speed_ms = float(ego_speed[index]) if ego_speed is not None else None

        if tracker is not None:
            tracker.step(
                stride / fps,
                raw_distance,
                nearest,
                path.cross_track_error_m if path else None,
                path.heading_error_rad if path else None,
                raw_curvature,
            )
            has_lead = tracker.has_lead
            distance = tracker.lead.value if has_lead else out["lead_distance_m"]
            stddev = tracker.lead.stddev if has_lead else None
            closing = (-tracker.lead.rate) if has_lead else None
            cross_track = (
                tracker.cross_track.value
                if tracker.cross_track.value is not None
                else (path.cross_track_error_m if path else 0.0)
            )
            heading = (
                tracker.heading.value
                if tracker.heading.value is not None
                else (path.heading_error_rad if path else 0.0)
            )
            curvature = (
                tracker.curvature.value
                if tracker.curvature.value is not None
                else raw_curvature
            )
        else:
            has_lead = detector_lead
            distance = out["lead_distance_m"]
            stddev = closing = None
            cross_track = path.cross_track_error_m if path else 0.0
            heading = path.heading_error_rad if path else 0.0
            curvature = raw_curvature

        radius = (1.0 / abs(curvature)) if (curvature and abs(curvature) > 1e-5) else None
        ttc = (distance / closing) if (has_lead and closing and closing > 0.3) else None
        acceleration = None
        if speed_ms is not None:
            acceleration = planner.acceleration(
                curvature or 0.0,
                speed_ms,
                bool(has_lead),
                closing or 0.0,
                distance if has_lead else 9999.0,
            )

        state = {
            "H": camera.H,
            "time_s": index / fps,
            "frame": index,
            "inference_ms": inference_s * 1000,
            "ego_speed_kmh": None if speed_ms is None else speed_ms * 3.6,
            "has_lead": has_lead,
            "lead_distance_m": distance,
            "lead_stddev_m": stddev,
            "closing_speed_ms": closing,
            "ttc_s": ttc,
            "acceleration_ms2": acceleration,
            "lead_confidence": out["lead_confidence"],
            "objects_in_frame": len(out["detections"]),
            "objects_in_corridor": corridor_count,
            "nearest_in_corridor_m": nearest,
            "curvature": curvature,
            "radius_m": radius,
            "cross_track_error_m": cross_track,
            "heading_error_rad": heading,
            "look_ahead_m": path.x_max if path else 0.0,
            "waypoints": len(points),
        }
        writer.write(render_frame(resized, out["detections"], path, state))

        rows.append({
            "frame": index,
            "time_s": round(index / fps, 3),
            "ego_speed_kmh": "" if speed_ms is None else round(speed_ms * 3.6, 1),
            "lead_present": int(bool(has_lead)),
            "lead_distance_m": round(distance, 2),
            "lead_distance_stddev_m": "" if stddev is None else round(stddev, 2),
            "lead_closing_speed_kmh": "" if closing is None else round(closing * 3.6, 1),
            "time_to_collision_s": "" if ttc is None else round(ttc, 2),
            "target_acceleration_ms2": (
                "" if acceleration is None else round(acceleration, 3)
            ),
            "lead_confidence": round(out["lead_confidence"], 4),
            "lead_distance_raw_m": round(out["lead_distance_m"], 2),
            "objects_in_frame": len(out["detections"]),
            "objects_in_corridor": corridor_count,
            "nearest_in_corridor_m": "" if nearest is None else round(nearest, 2),
            "path_curvature_1pm": "" if curvature is None else round(curvature, 6),
            "path_radius_m": "" if radius is None else round(radius, 1),
            "cross_track_error_m": round(cross_track, 3),
            "heading_error_deg": round(math.degrees(heading), 2),
            "look_ahead_m": round(path.x_max, 1) if path else "",
            "waypoints": len(points),
            "inference_ms": round(inference_s * 1000, 1),
        })

        done += 1
        if done % 25 == 0:
            remaining = (time.time() - started) / done * (planned - done)
            print(f"  {done}/{planned}  {index / fps:6.1f}s  lead={distance:6.1f} m  "
                  f"wp={len(points):2d} obj={len(out['detections']):2d}  "
                  f"~{remaining / 60:.1f} min left", flush=True)

    capture.release()
    writer.release()

    if not rows:
        raise SystemExit("no frames were processed — check --from-s / --to-s")

    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    timings = np.array([r["inference_ms"] for r in rows])
    print(f"\n{done} frames in {(time.time() - started) / 60:.1f} min, "
          f"median {np.median(timings):.0f} ms per frame "
          f"({1000 / np.median(timings):.2f} fps)")
    print(f"wrote {args.out}/annotated.mp4 and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
