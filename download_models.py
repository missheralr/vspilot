"""Download the three ONNX model files into ``weights/``. Run once.

No git required. The files come straight from raw.githubusercontent.com:
each is under 100 MB, so GitHub serves them directly rather than through
Git LFS. 114 MB in total.

Only the fp32 weights are fetched. int8 was measured on a CPU without VNNI
and came out *slower* than fp32 (1409 ms against 950 ms) because the int8
kernels are emulated, so there is no reason to download them by default.
"""

from __future__ import annotations

import os
import sys
import urllib.request

BASE_URL = (
    "https://raw.githubusercontent.com/autowarefoundation/"
    "autoware_vision_pilot/main/VisionPilot/modules/models/weights"
)

# Exact sizes, confirmed with an HTTP HEAD request. Used to verify the
# download completed rather than trusting that it did.
FILES = {
    "autodrive_fp32.onnx": 43015192,
    "autosteer_fp32.onnx": 33266514,
    "autospeed_fp32.onnx": 38015647,
}


def download(target_dir: str | None = None, log=print) -> str:
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(target_dir, exist_ok=True)

    total = sum(FILES.values())
    log(f"Downloading {len(FILES)} model files, {total / 1048576:.0f} MB, to {target_dir}")

    for name, expected_size in FILES.items():
        destination = os.path.join(target_dir, name)
        if os.path.isfile(destination) and os.path.getsize(destination) == expected_size:
            log(f"  {name}: already present, skipping")
            continue

        partial = destination + ".part"
        log(f"  {name}: downloading {expected_size / 1048576:.0f} MB ...")

        def progress(block, block_size, file_size, _name=name):
            done = block * block_size
            if block % 200 == 0:
                log(f"    {_name}  {min(done, file_size) / 1048576:6.0f} / "
                    f"{file_size / 1048576:.0f} MB")

        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", partial, progress)
        except Exception as exc:
            if os.path.isfile(partial):
                os.remove(partial)
            raise SystemExit(
                f"  Download failed: {exc}\n  Check the connection and run again."
            )

        actual = os.path.getsize(partial)
        if actual != expected_size:
            os.remove(partial)
            raise SystemExit(
                f"  {name} came down as {actual} bytes but should be "
                f"{expected_size}.\n  The file is incomplete and has been deleted. "
                "Run again."
            )
        os.replace(partial, destination)
        log(f"  {name}: done")

    log("Finished. This does not need running again.")
    return target_dir


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else None)
