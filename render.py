"""Overlay drawing: the ego path band and the information panel."""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration import NET_HEIGHT, NET_WIDTH

# --- ego path band ---------------------------------------------------------
# visual_interface.cpp:219-254 does NOT scatter dots. It fits a quadratic to
# the waypoints, samples it every 0.5 m, builds a band +-1 m wide around it,
# fills that, and blends with addWeighted(overlay, 0.35, img, 0.65).
PATH_HALF_WIDTH_M = 1.0
PATH_SAMPLE_STEP_M = 0.5
PATH_ALPHA = 0.35

# visual_interface.cpp:153-162, BGR. The band colour encodes the planner's
# demanded acceleration.
COLOUR_HARD_BRAKE = (93, 0, 255)
COLOUR_BRAKING = (0, 102, 255)
COLOUR_LIGHT_BRAKE = (0, 213, 255)
COLOUR_CRUISE = (174, 255, 0)

COLOUR_DETECTION = (120, 230, 120)
COLOUR_BACKGROUND = (28, 24, 20)
COLOUR_TEXT = (236, 240, 245)
COLOUR_MUTED = (150, 156, 164)
COLOUR_VALUE = (120, 200, 255)
COLOUR_ALERT = (90, 180, 255)
COLOUR_DISABLED = (110, 114, 120)
COLOUR_RULE = (62, 58, 52)

PANEL_WIDTH = 420
FONT = cv2.FONT_HERSHEY_SIMPLEX


def path_colour(acceleration: float | None):
    if acceleration is None:
        return COLOUR_CRUISE
    if acceleration < -5.0:
        return COLOUR_HARD_BRAKE
    if acceleration < -3.0:
        return COLOUR_BRAKING
    if acceleration < -1.0:
        return COLOUR_LIGHT_BRAKE
    return COLOUR_CRUISE


def acceleration_label(acceleration: float | None) -> str:
    if acceleration is None:
        return "n/a"
    if acceleration < -5.0:
        return "hard brake"
    if acceleration < -3.0:
        return "braking"
    if acceleration < -1.0:
        return "light brake"
    return "cruise"


def draw_path_band(image, path, H, acceleration=None) -> None:
    """Fill the +-1 m corridor around the fitted path and blend it in."""
    colour = path_colour(acceleration)
    world_to_pixel = np.linalg.inv(H)

    def project(x, y):
        q = world_to_pixel @ np.array([x, y, 1.0])
        if abs(q[2]) < 1e-9:
            return None
        return (int(round(q[0] / q[2])), int(round(q[1] / q[2])))

    left, right = [], []
    x = max(0.5, path.x_min)
    while x <= path.x_max + 1e-3:
        centre = path.lateral_at(x)
        pl = project(x, centre + PATH_HALF_WIDTH_M)
        pr = project(x, centre - PATH_HALF_WIDTH_M)
        if pl and pr:
            left.append(pl)
            right.append(pr)
        x += PATH_SAMPLE_STEP_M
    if len(left) < 2:
        return

    overlay = image.copy()
    h, w = image.shape[:2]
    for i in range(len(left) - 1):
        quad = np.array([left[i], right[i], right[i + 1], left[i + 1]], np.int32)
        inside = (
            (quad[:, 0] >= 0) & (quad[:, 0] < w) & (quad[:, 1] >= 0) & (quad[:, 1] < h)
        )
        if not inside.any():
            continue
        cv2.fillConvexPoly(overlay, quad, colour, cv2.LINE_AA)
    cv2.addWeighted(overlay, PATH_ALPHA, image, 1.0 - PATH_ALPHA, 0.0, image)
    cv2.polylines(image, [np.array(left, np.int32)], False, colour, 1, cv2.LINE_AA)
    cv2.polylines(image, [np.array(right, np.int32)], False, colour, 1, cv2.LINE_AA)


def _text(image, s, origin, scale=0.44, colour=COLOUR_TEXT, thickness=1):
    cv2.putText(image, s, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_panel(state: dict) -> np.ndarray:
    """Render the information panel that sits beside the video."""
    panel = np.full((NET_HEIGHT, PANEL_WIDTH, 3), COLOUR_BACKGROUND, np.uint8)
    cv2.line(panel, (0, 0), (0, NET_HEIGHT), (70, 66, 60), 1)
    width = PANEL_WIDTH

    def rule(y, heading=None):
        cv2.line(panel, (18, y), (width - 18, y), COLOUR_RULE, 1)
        if heading:
            _text(panel, heading, (18, y + 16), 0.36, COLOUR_MUTED)
            return y + 33  # must clear the heading or the first row collides
        return y + 10

    def row(y, label, value, colour=COLOUR_VALUE, scale=0.42):
        _text(panel, label, (18, y), 0.40, COLOUR_TEXT)
        (w, _), _ = cv2.getTextSize(value, FONT, scale, 1)
        _text(panel, value, (width - 18 - w, y), scale, colour)
        return y + 17

    _text(panel, "VISION  PILOT", (18, 26), 0.52, COLOUR_VALUE)
    _text(panel, "70mai A800S  |  perception + tracking", (18, 42), 0.33, COLOUR_MUTED)

    # --- headline numbers --------------------------------------------------
    y = rule(48)
    _text(panel, "EGO SPEED", (18, y + 8), 0.34, COLOUR_MUTED)
    if state["ego_speed_kmh"] is None:
        _text(panel, "n/a", (18, y + 38), 0.85, COLOUR_DISABLED, 2)
    else:
        value = f"{state['ego_speed_kmh']:.0f}"
        _text(panel, value, (18, y + 40), 1.30, COLOUR_TEXT, 2)
        (w, _), _ = cv2.getTextSize(value, FONT, 1.30, 2)
        _text(panel, "km/h", (22 + w, y + 40), 0.42, COLOUR_MUTED)

    _text(panel, "LEAD GAP", (width // 2 + 10, y + 8), 0.34, COLOUR_MUTED)
    if state["has_lead"]:
        value = f"{state['lead_distance_m']:.1f}"
        _text(panel, value, (width // 2 + 10, y + 40), 1.30, COLOUR_TEXT, 2)
        (w, _), _ = cv2.getTextSize(value, FONT, 1.30, 2)
        _text(panel, "m", (width // 2 + 14 + w, y + 40), 0.42, COLOUR_MUTED)
    else:
        _text(panel, "none", (width // 2 + 10, y + 38), 0.85, COLOUR_DISABLED, 2)
    y += 46

    # --- lead vehicle ------------------------------------------------------
    y = rule(y, "LEAD VEHICLE")
    if state["has_lead"]:
        closing = state["closing_speed_ms"]
        y = row(
            y,
            "closing speed",
            "n/a" if closing is None else f"{closing * 3.6:+.1f} km/h",
            COLOUR_ALERT if (closing or 0) > 0.5 else COLOUR_VALUE,
        )
        ttc = state["ttc_s"]
        y = row(
            y,
            "time to collision",
            "safe" if ttc is None else f"{ttc:.1f} s",
            COLOUR_ALERT if (ttc is not None and ttc < 4.0) else COLOUR_VALUE,
        )
        sd = state["lead_stddev_m"]
        y = row(y, "gap uncertainty", "n/a" if sd is None else f"+- {sd:.1f} m")
    else:
        y = row(y, "detector confidence", f"{state['lead_confidence']:.2f}")
        y += 34

    # --- longitudinal plan -------------------------------------------------
    y = rule(y, "LONGITUDINAL PLAN")
    acc = state["acceleration_ms2"]
    if acc is None:
        y = row(y, "target acceleration", "needs ego speed", COLOUR_DISABLED, 0.38)
    else:
        colour = path_colour(acc)
        y = row(y, "target acceleration", f"{acc:+.2f} m/s2", colour)
        y = row(y, "state", acceleration_label(acc), colour)

    # --- path --------------------------------------------------------------
    y = rule(y, "PATH")
    if state["curvature"] is not None:
        radius = state["radius_m"]
        y = row(y, "curvature", f"{state['curvature']:+.4f} 1/m")
        y = row(y, "radius", "straight" if radius is None else f"{radius:,.0f} m")
        y = row(y, "cross-track error", f"{state['cross_track_error_m']:+.2f} m")
        y = row(
            y,
            "heading error",
            f"{math.degrees(state['heading_error_rad']):+.1f} deg",
        )
    else:
        y = row(y, "path fit", "not enough points", COLOUR_DISABLED, 0.38)
        y += 34

    # --- traffic -----------------------------------------------------------
    y = rule(y, "TRAFFIC")
    y = row(y, "objects in frame", f"{state['objects_in_frame']}")
    y = row(y, "in ego corridor", f"{state['objects_in_corridor']}")
    nearest = state["nearest_in_corridor_m"]
    y = row(
        y,
        "nearest in corridor",
        "none" if nearest is None else f"{nearest:.1f} m",
    )

    # --- footer ------------------------------------------------------------
    cv2.line(panel, (18, NET_HEIGHT - 60), (width - 18, NET_HEIGHT - 60), COLOUR_RULE, 1)
    _text(panel, f"waypoints {state['waypoints']}/64", (18, NET_HEIGHT - 42), 0.36,
          COLOUR_MUTED)
    _text(panel, f"look-ahead {state['look_ahead_m']:.0f} m", (width - 150,
          NET_HEIGHT - 42), 0.36, COLOUR_MUTED)
    _text(panel, f"t = {state['time_s']:.1f} s   frame {state['frame']}",
          (18, NET_HEIGHT - 22), 0.36, COLOUR_MUTED)
    _text(panel, f"{state['inference_ms']:.0f} ms", (width - 150, NET_HEIGHT - 22),
          0.36, COLOUR_MUTED)
    return panel


def render_frame(image, detections, path, state) -> np.ndarray:
    """Draw the band and boxes on the frame, then append the panel."""
    canvas = image.copy()
    if path is not None:
        draw_path_band(canvas, path, state["H"], state.get("acceleration_ms2"))
    for box, score in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOUR_DETECTION, 2)
        _text(canvas, f"{score:.2f}", (x1, max(12, y1 - 6)), 0.42, COLOUR_DETECTION)
    return np.hstack([canvas, draw_panel(state)])
