"""Projecting model outputs onto the road plane.

All world coordinates follow the convention stated at
``lateral_fusion.cpp:200``: ``x`` forward in metres, ``y`` lateral in metres,
positive to the left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from calibration import NET_HEIGHT, NET_WIDTH
from models import NUM_WAYPOINTS, WAYPOINT_CONFIDENCE


@dataclass
class PathFit:
    """A quadratic ``y = a x^2 + b x + c`` fitted to the AutoSteer waypoints."""

    a: float
    b: float
    c: float
    x_min: float
    x_max: float
    cross_track_error_m: float
    heading_error_rad: float

    @property
    def curvature(self) -> float:
        """Signed curvature in 1/m. Positive turns left."""
        return 2.0 * self.a

    @property
    def radius_m(self) -> float | None:
        k = abs(self.curvature)
        return (1.0 / k) if k > 1e-5 else None

    def lateral_at(self, x: float) -> float:
        return self.a * x * x + self.b * x + self.c


def waypoints_to_world(
    waypoint_x: np.ndarray, waypoint_conf: np.ndarray, H: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Port of ``LateralFusion::waypoints`` (``lateral_fusion.cpp:185-197``).

    Returns ``(u, v, x_forward, y_lateral)`` for every waypoint that clears
    the confidence threshold and lands in a sensible forward range.
    """
    points = []
    for i in range(NUM_WAYPOINTS):
        if waypoint_conf[i] < WAYPOINT_CONFIDENCE:
            continue
        v = i * (NET_HEIGHT - 1.0) / (NUM_WAYPOINTS - 1)
        u = float(waypoint_x[i]) * NET_WIDTH
        q = H @ np.array([u, v, 1.0])
        if abs(q[2]) < 1e-12:
            continue
        x_forward, y_lateral = q[0] / q[2], q[1] / q[2]
        if 0.5 <= x_forward <= 120.0:
            points.append((u, v, float(x_forward), float(y_lateral)))
    return points


def fit_path(points) -> PathFit | None:
    """Fit the ego path, following ``LateralFusion::fit_ransac``.

    Cross-track error and heading are evaluated at ``x_min`` — the nearest
    waypoint actually observed — not at ``x = 0``. ``lateral_fusion.cpp:331``
    spells out why: evaluating at zero extrapolates back to where there is no
    data.
    """
    if len(points) < 4:
        return None
    x = np.array([p[2] for p in points])
    y = np.array([p[3] for p in points])
    near = x <= 60.0  # waypoints beyond this are very noisy
    if near.sum() < 4:
        return None
    x, y = x[near], y[near]
    a, b, c = (float(v) for v in np.polyfit(x, y, 2))
    x_min, x_max = float(x.min()), float(x.max())
    return PathFit(
        a=a,
        b=b,
        c=c,
        x_min=x_min,
        x_max=x_max,
        cross_track_error_m=a * x_min * x_min + b * x_min + c,
        heading_error_rad=math.atan(2.0 * a * x_min + b),
    )


def objects_in_corridor(
    detections, H: np.ndarray, half_width_m: float = 1.8
) -> tuple[int, float | None]:
    """Count detections whose footprint falls inside the ego corridor.

    The box bottom is projected onto the road plane, which is how
    ``longitudinal_fusion.cpp:415`` derives a distance from an AutoSpeed box.
    This is the number worth trusting: the raw detection count includes
    roadside clutter such as bushes and poles.

    Returns ``(count, nearest_distance_m)``.
    """
    count, nearest = 0, None
    for box, _score in detections:
        u = (box[0] + box[2]) / 2.0
        q = H @ np.array([u, box[3], 1.0])
        if abs(q[2]) < 1e-12:
            continue
        x_forward, y_lateral = q[0] / q[2], q[1] / q[2]
        if not (0.5 <= x_forward <= 150.0) or abs(y_lateral) > half_width_m:
            continue
        count += 1
        if nearest is None or x_forward < nearest:
            nearest = float(x_forward)
    return count, nearest
