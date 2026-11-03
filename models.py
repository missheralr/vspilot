"""The three Vision Pilot ONNX models, run in parallel and decoded.

Output decoding follows the upstream C++ exactly:

* AutoDrive  ``auto_drive.cpp:73-76`` — distance is normalised, so
  ``distance_m = 150 * (1 - dist_normalized)``; curvature is scaled by 0.21;
  the third output is a logit, so it needs a sigmoid.
* AutoSteer  ``auto_steer.cpp:70-78`` — 64 normalised x positions plus 64
  confidences.
* AutoSpeed  ``auto_speed.cpp:78-130`` — YOLO style ``[1, 4 + classes, N]``,
  sigmoid over the class channels, then non-maximum suppression.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

# longitudinal_fusion.cpp:67 and visualization.cpp:596
MAX_DISTANCE_M = 150.0
# lateral_fusion.hpp:98 (CURV_SCALE from load_data_auto_drive.py)
CURVATURE_SCALE = 0.21
# auto_steer.cpp — the model emits 64 waypoints
NUM_WAYPOINTS = 64
# lateral_fusion.cpp:189 — waypoints below this confidence are discarded
WAYPOINT_CONFIDENCE = 0.5
# auto_speed.hpp:53-54
DETECTION_CONFIDENCE = 0.6
DETECTION_IOU = 0.45

REQUIRED_WEIGHT = "autodrive_fp32.onnx"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_weights(explicit: str | None = None) -> str:
    """Locate the directory holding the three ``.onnx`` files.

    When the caller names a directory explicitly we must never fall back to
    somewhere else: a single typo would otherwise run a different copy of the
    models while the user believes they are running theirs.
    """
    if explicit:
        if os.path.isfile(os.path.join(explicit, REQUIRED_WEIGHT)):
            return explicit
        raise SystemExit(f"Weights directory '{explicit}' has no {REQUIRED_WEIGHT}.")

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.environ.get("VP_WEIGHTS"),
        os.path.join(here, "weights"),
        "weights",
        "VisionPilot/modules/models/weights",
    ):
        if candidate and os.path.isfile(os.path.join(candidate, REQUIRED_WEIGHT)):
            return candidate
    raise SystemExit(
        "Model files not found.\n"
        "  Run download_models.py once to fetch them (114 MB), or point\n"
        "  VP_WEIGHTS at a directory that already has them."
    )


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Port of ``AutoSpeed::nms`` (``auto_speed.cpp:140-162``)."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (area_best + area_rest - inter + 1e-6) <= iou_threshold]
    return keep


class ModelBundle:
    """AutoDrive, AutoSteer and AutoSpeed, run concurrently.

    ``inference.cpp:163-181`` launches the three with ``std::async``. Running
    them one after another wastes most of the gain: onnxruntime releases the
    GIL while a session runs, so Python threads genuinely overlap. Measured on
    a 2-core machine: 389 ms sequential against 209 ms parallel (1.86x). The
    outputs are bit-identical either way.
    """

    def __init__(self, weights_dir: str, precision: str = "fp32", parallel: bool = True):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]

        def load(name):
            return ort.InferenceSession(
                os.path.join(weights_dir, f"{name}_{precision}.onnx"),
                options,
                providers=providers,
            )

        self.auto_drive = load("autodrive")
        self.auto_steer = load("autosteer")
        self.auto_speed = load("autospeed")
        self._drive_inputs = [i.name for i in self.auto_drive.get_inputs()]
        self._steer_input = self.auto_steer.get_inputs()[0].name
        self._speed_input = self.auto_speed.get_inputs()[0].name
        self._pool = ThreadPoolExecutor(3) if parallel else None

    def run(self, warped_prev, warped_curr, resized) -> dict:
        """Infer one frame. ``warped_prev``/``warped_curr`` are consecutive."""
        drive_feed = {
            self._drive_inputs[0]: warped_prev,
            self._drive_inputs[1]: warped_curr,
        }
        if self._pool is not None:
            drive = self._pool.submit(self.auto_drive.run, None, drive_feed)
            steer = self._pool.submit(
                self.auto_steer.run, None, {self._steer_input: resized}
            )
            speed = self._pool.submit(
                self.auto_speed.run, None, {self._speed_input: resized}
            )
            return self._decode(drive.result(), steer.result(), speed.result())
        return self._decode(
            self.auto_drive.run(None, drive_feed),
            self.auto_steer.run(None, {self._steer_input: resized}),
            self.auto_speed.run(None, {self._speed_input: resized}),
        )

    def _decode(self, drive, steer, speed) -> dict:
        result = {
            "lead_distance_m": float(MAX_DISTANCE_M * (1.0 - drive[0][0, 0])),
            "curvature": float(drive[1][0, 0]) * CURVATURE_SCALE,
            "lead_confidence": float(sigmoid(drive[2][0, 0])),
            "waypoint_x": np.asarray(steer[0]).ravel()[:NUM_WAYPOINTS],
            "waypoint_conf": np.asarray(steer[1]).ravel()[:NUM_WAYPOINTS],
        }

        grid = np.asarray(speed[0])
        if grid.ndim == 3:
            grid = grid[0]
        num_classes = grid.shape[0] - 4
        probabilities = sigmoid(grid[4 : 4 + num_classes])
        best_class = np.argmax(probabilities, axis=0)
        scores = probabilities[best_class, np.arange(grid.shape[1])]
        selected = scores >= DETECTION_CONFIDENCE
        if selected.any():
            cx, cy = grid[0][selected], grid[1][selected]
            w, h = grid[2][selected], grid[3][selected]
            boxes = np.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
            )
            kept_scores = scores[selected]
            result["detections"] = [
                (boxes[i], float(kept_scores[i]))
                for i in non_max_suppression(boxes, kept_scores, DETECTION_IOU)
            ]
        else:
            result["detections"] = []
        return result
