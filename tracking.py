"""State carried between frames, and the IDM longitudinal planner.

Without tracking every frame is independent: the numbers jitter badly and
there is no way to know whether the lead vehicle is closing or pulling away.
Upstream solves this with particle filters
(``safety_guardian/fusion``, about 1100 lines). This module uses
constant-velocity Kalman filters instead: same purpose, far less code, and
they still produce the standard deviation that upstream exposes as
``distance_stddev_m``.

Measured on a real 180 s clip, mean absolute change per frame:

=====================  ==========  ========  ========
quantity               untracked   tracked   reduction
=====================  ==========  ========  ========
lead distance            2.40 m     0.54 m      78%
cross-track error        0.36 m     0.11 m      69%
heading error            8.15 deg   4.44 deg    45%
=====================  ==========  ========  ========
"""

from __future__ import annotations

import math

import numpy as np


class ConstantVelocityFilter:
    """Two-state Kalman filter over ``[value, rate of change]``.

    Args:
        process_noise: higher follows the measurements faster and noisier.
        measurement_noise: higher trusts each measurement less.
        gate: reject a measurement this many predicted standard deviations
            away. Without a gate one bad detection drags the estimate with it.
    """

    def __init__(self, process_noise=1.0, measurement_noise=4.0, gate=4.0):
        self.q = process_noise
        self.r = measurement_noise
        self.gate = gate
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.consecutive_rejections = 0

    def start(self, measurement: float, rate: float = 0.0) -> None:
        self.x = np.array([float(measurement), float(rate)])
        self.P = np.array([[self.r, 0.0], [0.0, 25.0]])
        self.consecutive_rejections = 0

    def predict(self, dt: float) -> None:
        if self.x is None:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[0.5 * dt * dt], [dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (G @ G.T) * self.q

    def update(self, measurement: float, measurement_noise: float | None = None) -> bool:
        """Fold in one measurement. Returns False if the gate rejected it."""
        if self.x is None:
            self.start(measurement)
            return True
        r = self.r if measurement_noise is None else measurement_noise
        innovation_cov = self.P[0, 0] + r
        innovation = float(measurement) - self.x[0]
        if abs(innovation) > self.gate * math.sqrt(innovation_cov):
            self.consecutive_rejections += 1
            return False
        gain = self.P[:, 0] / innovation_cov
        self.x = self.x + gain * innovation
        self.P = self.P - np.outer(gain, self.P[0, :])
        self.consecutive_rejections = 0
        return True

    @property
    def value(self) -> float | None:
        return None if self.x is None else float(self.x[0])

    @property
    def rate(self) -> float | None:
        return None if self.x is None else float(self.x[1])

    @property
    def stddev(self) -> float | None:
        return None if self.P is None else float(math.sqrt(max(0.0, self.P[0, 0])))

    def reset(self) -> None:
        self.x = self.P = None
        self.consecutive_rejections = 0


class SceneTracker:
    """All per-frame state that has to survive into the next frame."""

    MAX_MISSES = 5  # drop the lead track after this many frames with no measurement

    def __init__(self):
        self.lead = ConstantVelocityFilter(2.0, 6.0, gate=4.0)
        self.cross_track = ConstantVelocityFilter(0.5, 0.25, gate=5.0)
        self.heading = ConstantVelocityFilter(0.5, 0.01, gate=5.0)
        self.curvature = ConstantVelocityFilter(1e-5, 1e-4, gate=6.0)
        self.misses = 99

    def step(
        self,
        dt: float,
        lead_from_model: float | None,
        lead_from_geometry: float | None,
        cross_track_m: float | None,
        heading_rad: float | None,
        curvature: float | None,
    ) -> None:
        """Advance one frame.

        Two independent distance measurements are folded in, which is what
        ``LongitudinalFusion`` does: AutoDrive's own estimate, and the
        distance implied by projecting the detection box bottom onto the road.
        The geometric one is given lower measurement noise because it comes
        from an explicit projection rather than a regression head.
        """
        for f in (self.lead, self.cross_track, self.heading, self.curvature):
            f.predict(dt)

        if lead_from_model is None and lead_from_geometry is None:
            self.misses += 1
            if self.misses > self.MAX_MISSES:
                self.lead.reset()
        else:
            if self.lead.x is None:
                self.lead.start(
                    lead_from_model if lead_from_model is not None else lead_from_geometry
                )
            accepted = False
            if lead_from_model is not None:
                accepted |= self.lead.update(lead_from_model, measurement_noise=6.0)
            if lead_from_geometry is not None:
                accepted |= self.lead.update(lead_from_geometry, measurement_noise=3.0)

            if accepted:
                self.misses = 0
            else:
                self.misses += 1
                # Several rejections in a row means a different vehicle is now
                # in front, not that the measurements are bad. Re-seed.
                if self.lead.consecutive_rejections >= 3:
                    self.lead.start(
                        lead_from_geometry
                        if lead_from_geometry is not None
                        else lead_from_model
                    )
                    self.misses = 0
                elif self.misses > self.MAX_MISSES:
                    self.lead.reset()

        if cross_track_m is not None:
            self.cross_track.update(cross_track_m)
            self.heading.update(heading_rad)
        if curvature is not None:
            self.curvature.update(curvature)

    @property
    def has_lead(self) -> bool:
        return self.lead.x is not None and self.misses <= self.MAX_MISSES


class LongitudinalPlanner:
    """Port of ``LongitudinalPlanner::compute_acceleration``.

    Defaults are taken verbatim from ``longitudinal_planning.hpp:9-17``.

    Note that IDM has no lower bound, and neither does upstream — it returns
    the raw value. In a near-collision geometry it can demand something like
    -146 m/s2. That is the algorithm behaving as designed, not a bug.

    The default 1.5 s headway is tuned for Western highway driving. On a real
    Vietnamese city clip the measured headway was 1.09 s with 66% of frames
    below 1.5 s, so IDM asked for braking almost continuously. See the README.
    """

    def __init__(
        self,
        speed_limit_ms: float,
        a: float = 1.5,
        b: float = 3.0,
        headway_s: float = 1.5,
        min_gap_m: float = 2.0,
        delta: float = 4.0,
        mu: float = 0.2,
        g: float = 9.81,
    ):
        self.speed_limit = speed_limit_ms
        self.a, self.b = a, b
        self.headway_s, self.min_gap_m = headway_s, min_gap_m
        self.delta, self.mu, self.g = delta, mu, g

    def acceleration(
        self,
        curvature: float,
        ego_speed_ms: float,
        has_lead: bool,
        closing_speed_ms: float,
        gap_m: float,
    ) -> float:
        """``closing_speed_ms`` is positive when the gap is shrinking."""
        k = abs(curvature) if curvature else 0.0
        curve_limit = math.sqrt(self.mu * self.g / k) if k > 1e-9 else float("inf")
        speed_limit = min(self.speed_limit, curve_limit)

        dynamic = (ego_speed_ms * closing_speed_ms) / (2.0 * math.sqrt(self.a * self.b))
        desired_gap = self.min_gap_m + max(
            0.0, ego_speed_ms * self.headway_s + dynamic
        )
        gap = max(0.5, gap_m)
        free_road = (
            (ego_speed_ms / speed_limit) ** self.delta if speed_limit > 1e-9 else 1.0
        )
        interaction = (desired_gap / gap) ** 2.0 if has_lead else 0.0
        return self.a * (1.0 - free_road - interaction)
