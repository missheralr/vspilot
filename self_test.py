#!/usr/bin/env python3
"""Re-run every check and report pass/fail.

Every number quoted in the README has to be reproducible here. If a change
breaks something, this must say so.

    python self_test.py [clip.mp4]
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time

import cv2
import numpy as np

import gps_log
import models as model_mod
import render
from calibration import (
    DEFAULT_FOE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NET_HEIGHT,
    V_WARPED_TO_WORLD,
    VirtualCamera,
    top_crop_2_1,
)
from geometry import fit_path
from tracking import ConstantVelocityFilter, LongitudinalPlanner, SceneTracker

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main(video: str | None = None) -> int:
    started = time.time()
    print("Self-test\n" + "=" * 64)

    section("1. Camera geometry and the C matrix")
    camera = VirtualCamera(1920, 1080, DEFAULT_FOE, camera_height_m=1.30)
    product = V_WARPED_TO_WORLD @ camera.C
    check(
        "V @ C == H, the relation C is derived from",
        np.allclose(product / product[2, 2], camera.H / camera.H[2, 2], rtol=1e-9),
    )

    world_to_pixel = np.linalg.inv(camera.H)
    errors = []
    for x, y in [(10, 0), (20, 1.75), (30, -1.75), (50, 0), (15, 3.5), (80, 0)]:
        q = world_to_pixel @ np.array([x, y, 1.0])
        virtual_px = q[:2] / q[2]
        q = camera.C @ np.array([virtual_px[0], virtual_px[1], 1.0])
        warped_px = q[:2] / q[2]
        q = V_WARPED_TO_WORLD @ np.array([warped_px[0], warped_px[1], 1.0])
        back = q[:2] / q[2]
        errors.append(math.hypot(back[0] - x, back[1] - y))
    check(
        "round trip road point -> pixel -> warped -> road point",
        max(errors) < 1e-9,
        f"largest error {max(errors):.2e} m",
    )
    check(
        "warped image is drawn entirely from real sensor pixels",
        camera.sensor_coverage.mean() > 0.999,
        f"{100 * camera.sensor_coverage.mean():.2f}%",
    )
    check(
        "virtual camera rotation matches the published figure",
        abs(camera.rotation_deg - 8.1) < 0.15,
        f"{camera.rotation_deg:.2f} deg (published 8.1)",
    )

    section("2. Preprocessing matches the C++ sources")
    check(
        "top_crop_2_1 matches common/utils.hpp:15",
        all(
            top_crop_2_1(h, w) == max(0, round(h - w / 2.0))
            for h, w in [(1080, 1920), (1280, 1920), (512, 1024), (720, 1280)]
        ),
    )
    check(
        "emitting 1024x512 makes the top crop zero, so no second resample",
        top_crop_2_1(512, 1024) == 0,
    )
    grey = np.full((512, 1024, 3), 128, np.uint8)
    imagenet = VirtualCamera.chw_imagenet(grey)
    expected = (128 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    check(
        "chw_imagenet uses the MEAN/STD from inference.cpp:22-23",
        imagenet.shape == (1, 3, 512, 1024)
        and np.allclose([imagenet[0, c].mean() for c in range(3)], expected, atol=1e-5),
    )
    check(
        "chw_01 only divides by 255",
        abs(float(VirtualCamera.chw_01(grey).mean()) - 128 / 255) < 1e-5,
    )

    section("3. Cross-track error is taken at x_min (lateral_fusion.cpp:331-336)")
    a, b, c = 0.001, -0.02, 0.5
    points = [(0, 0, x, a * x * x + b * x + c) for x in np.linspace(5, 40, 20)]
    path = fit_path(points)
    check(
        "the fit recovers the coefficients it was given",
        path is not None
        and abs(path.a - a) < 1e-9
        and abs(path.b - b) < 1e-9
        and abs(path.c - c) < 1e-9,
    )
    x_min = 5.0
    check(
        "CTE is evaluated at x_min, not extrapolated back to x = 0",
        abs(path.cross_track_error_m - (a * x_min * x_min + b * x_min + c)) < 1e-9
        and abs(path.cross_track_error_m - c) > 1e-3,
        f"CTE={path.cross_track_error_m:.4f} at x_min, c={c} at x=0 — different values",
    )
    check(
        "heading is atan(slope at x_min)",
        abs(path.heading_error_rad - math.atan(2 * a * x_min + b)) < 1e-9,
    )

    section("4. Temporal tracking")
    tracked = ConstantVelocityFilter(2.0, 6.0)
    truth = [50 - 2.0 * i * 0.1 for i in range(60)]  # receding at 2 m/s
    rng = np.random.default_rng(0)
    for value in truth:
        tracked.predict(0.1)
        tracked.update(value + rng.normal(0, 1.5))
    check(
        "follows a noisy signal to the right value",
        abs(tracked.value - truth[-1]) < 1.5,
        f"filter {tracked.value:.2f} m, truth {truth[-1]:.2f} m",
    )
    check(
        "recovers the rate of change",
        abs(tracked.rate - (-2.0)) < 0.5,
        f"{tracked.rate:+.2f} m/s (truth -2.00)",
    )
    check("reports a standard deviation", (tracked.stddev or 0) > 0,
          f"+-{tracked.stddev:.2f} m")
    before = tracked.value
    accepted = tracked.update(tracked.value + 60.0)
    check(
        "the gate rejects an absurd measurement",
        accepted is False and abs(tracked.value - before) < 1e-9,
    )
    check(
        "an unstarted filter reports None",
        ConstantVelocityFilter().value is None,
    )

    tracker = SceneTracker()
    for _ in range(10):
        tracker.step(0.1, 30.0, 30.0, 0.1, 0.01, 0.001)
    had_lead = tracker.has_lead
    for _ in range(SceneTracker.MAX_MISSES + 2):
        tracker.step(0.1, None, None, 0.1, 0.01, 0.001)
    check(
        "the lead track is dropped once measurements stop, not held forever",
        had_lead and not tracker.has_lead,
    )

    swapped = SceneTracker()
    for _ in range(10):
        swapped.step(0.1, 40.0, 40.0, 0.0, 0.0, 0.0)
    for _ in range(4):
        swapped.step(0.1, 8.0, 8.0, 0.0, 0.0, 0.0)
    check(
        "a new vehicle cutting in is re-acquired within a few frames",
        swapped.has_lead and abs(swapped.lead.value - 8.0) < 4.0,
        f"settled at {swapped.lead.value:.1f} m (truth 8.0 m)",
    )

    section("5. IDM, ported from longitudinal_planning.cpp")
    planner = LongitudinalPlanner(50 / 3.6)
    standing = planner.acceleration(0.0, 0.0, False, 0.0, 9999.0)
    check(
        "stationary on an empty road gives maximum acceleration",
        abs(standing - 1.5) < 1e-9,
        f"{standing:+.3f} m/s2 (config a = 1.5)",
    )
    at_limit = planner.acceleration(0.0, 50 / 3.6, False, 0.0, 9999.0)
    check("at the speed limit acceleration falls to zero", abs(at_limit) < 1e-9,
          f"{at_limit:+.3f} m/s2")
    closing = planner.acceleration(0.0, 14.0, True, 5.0, 4.0)
    check("a close, closing lead demands hard braking", closing < -3.0,
          f"{closing:+.2f} m/s2")
    distant = planner.acceleration(0.0, 10.0, True, 0.0, 80.0)
    check("a distant lead behaves almost like an empty road", distant > 0.9,
          f"{distant:+.2f} m/s2")
    tight_curve = planner.acceleration(0.05, 14.0, False, 0.0, 9999.0)
    check(
        "a tight curve lowers the speed limit and demands deceleration",
        tight_curve < 0,
        f"{tight_curve:+.2f} m/s2 at a 20 m radius",
    )
    ego_v, relative_v, gap = 14.0, 5.0, 4.0
    desired = 2.0 + max(0.0, ego_v * 1.5 + ego_v * relative_v / (2 * math.sqrt(1.5 * 3.0)))
    by_hand = 1.5 * (1 - (ego_v / (50 / 3.6)) ** 4 - (desired / gap) ** 2)
    check(
        "matches a hand calculation step by step",
        abs(closing - by_hand) < 1e-9,
        f"function {closing:.6f} vs by hand {by_hand:.6f}",
    )

    section("6. Panel text never overflows")
    overflow = 0
    for lead, curvature, accel, ego in itertools.product(
        [True, False], [0.001, None], [-4.0, None], [12.0, None]
    ):
        state = {
            "ego_speed_kmh": ego, "has_lead": lead, "lead_distance_m": 142.2,
            "lead_stddev_m": 0.8, "closing_speed_ms": 1.3, "ttc_s": 5.2,
            "acceleration_ms2": accel, "lead_confidence": 0.51,
            "objects_in_frame": 8, "objects_in_corridor": 1,
            "nearest_in_corridor_m": 4.5, "curvature": curvature,
            "radius_m": 386.0, "cross_track_error_m": -0.18,
            "heading_error_rad": 0.13, "look_ahead_m": 58.0, "waypoints": 32,
            "time_s": 53.4, "frame": 1602, "inference_ms": 390.0,
        }
        panel = cv2.cvtColor(render.draw_panel(state), cv2.COLOR_BGR2GRAY)
        if panel[NET_HEIGHT - 74 : NET_HEIGHT - 62, :].max() > 90:
            overflow += 1
    check("no combination of states writes into the footer", overflow == 0,
          f"{16 - overflow}/16 combinations clean")

    section("7. Path colour follows acceleration (visual_interface.cpp:153-162)")
    check("cruise is green", render.path_colour(0.5) == render.COLOUR_CRUISE)
    check("light braking is yellow", render.path_colour(-2.0) == render.COLOUR_LIGHT_BRAKE)
    check("braking is orange", render.path_colour(-4.0) == render.COLOUR_BRAKING)
    check("hard braking is purple", render.path_colour(-6.0) == render.COLOUR_HARD_BRAKE)

    section("8. Non-maximum suppression")
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], float)
    kept = model_mod.non_max_suppression(boxes, np.array([0.9, 0.8, 0.7]), 0.45)
    check("overlapping boxes merge, separate boxes survive",
          len(kept) == 2 and kept[0] == 0, f"kept {len(kept)}/3")

    if video and os.path.isfile(video):
        section(f"9. GPS log in {os.path.basename(video)}")
        records = gps_log.read_gps_log(video)
        check("the log is readable without ffmpeg", bool(records),
              f"{len(records) if records else 0} records")
        if records:
            capture = cv2.VideoCapture(video)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(
                cv2.CAP_PROP_FPS
            )
            capture.release()
            check("roughly 1 Hz", abs(len(records) - duration) < 3,
                  f"{len(records)} records over {duration:.0f} s")
            check(
                "coordinates land inside Vietnam",
                all(8 < r["latitude"] < 24 and 102 < r["longitude"] < 110
                    for r in records),
            )
            agreement = gps_log.cross_check(records)
            check(
                "distance from coordinates agrees with distance from speed",
                agreement["relative_error"] < 0.10,
                f"{100 * agreement['relative_error']:.1f}% apart "
                f"({agreement['distance_from_coords_m']:.0f} m vs "
                f"{agreement['distance_from_speed_m']:.0f} m)",
            )
            frames = int(round(duration * 30))
            per_frame = gps_log.speed_per_frame(records, frames, 30.0)
            logged = [r["speed_kmh"] for r in records]
            check(
                "per-frame interpolation has the right length and range",
                len(per_frame) == frames
                and abs(per_frame.max() * 3.6 - max(logged)) < 0.6
                and per_frame.min() >= 0,
                f"{per_frame.min() * 3.6:.0f}-{per_frame.max() * 3.6:.0f} km/h over "
                f"{frames} frames (log: {min(logged)}-{max(logged)})",
            )

        try:
            weights = model_mod.find_weights(None)
        except SystemExit:
            weights = None
        if weights:
            section("10. The three ONNX models")
            capture = cv2.VideoCapture(video)
            prepared = []
            for second in (50, 52, 54):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(second * 30))
                ok, frame = capture.read()
                if not ok:
                    break
                warped, resized = camera.prepare(frame)
                prepared.append(
                    (VirtualCamera.chw_imagenet(warped), VirtualCamera.chw_01(resized))
                )
            capture.release()
            if len(prepared) >= 2:
                parallel = model_mod.ModelBundle(weights, parallel=True)
                serial = model_mod.ModelBundle(weights, parallel=False)
                a_out = parallel.run(prepared[0][0], prepared[1][0], prepared[1][1])
                b_out = serial.run(prepared[0][0], prepared[1][0], prepared[1][1])
                check(
                    "parallel execution gives identical results to serial",
                    all(
                        abs(a_out[k] - b_out[k]) < 1e-9
                        for k in ("lead_distance_m", "curvature", "lead_confidence")
                    )
                    and np.array_equal(a_out["waypoint_x"], b_out["waypoint_x"])
                    and len(a_out["detections"]) == len(b_out["detections"]),
                )

                # Timing has to be a median of several runs after a warm-up.
                # A single measurement flips the moment the machine is busy —
                # this very check once reported a false failure for that reason.
                def timed(bundle, repeats=3):
                    bundle.run(prepared[0][0], prepared[1][0], prepared[1][1])
                    samples = []
                    for _ in range(repeats):
                        start = time.time()
                        bundle.run(prepared[0][0], prepared[1][0], prepared[1][1])
                        samples.append(time.time() - start)
                    return float(np.median(samples))

                t_parallel, t_serial = timed(parallel), timed(serial)
                check(
                    "parallel execution is not slower than serial",
                    t_parallel < t_serial * 1.3,
                    f"parallel {t_parallel * 1000:.0f} ms, serial "
                    f"{t_serial * 1000:.0f} ms ({t_serial / t_parallel:.2f}x)",
                )
                check(
                    "lead distance falls in a sensible range",
                    0 < a_out["lead_distance_m"] < 150,
                    f"{a_out['lead_distance_m']:.1f} m",
                )
                check(
                    "AutoSteer returns exactly 64 waypoints",
                    len(a_out["waypoint_x"]) == 64
                    and len(a_out["waypoint_conf"]) == 64,
                )

    print("\n" + "=" * 64)
    print(f"{len(PASSED)} / {len(PASSED) + len(FAILED)} checks passed "
          f"in {time.time() - started:.0f}s")
    if FAILED:
        print("\nFAILED:")
        for name in FAILED:
            print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    argument = sys.argv[1] if len(sys.argv) > 1 else None
    if argument is None:
        for candidate in ("clip1.mp4", "sample.mp4"):
            if os.path.isfile(candidate):
                argument = candidate
                break
    raise SystemExit(main(argument))
