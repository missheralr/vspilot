"""Camera model and the two input images Vision Pilot actually consumes.

Vision Pilot does not take one image. Reading
``modules/sensing/image_preprocessing/src/image_preprocessor.cpp`` and
``modules/models/src/inference.cpp`` shows it needs two, normalised
differently:

===========  ======================  ===========================
image        consumed by             normalisation
===========  ======================  ===========================
``warped``   AutoDrive               ImageNet mean/std
``resized``  AutoSteer, AutoSpeed    divide by 255 only
===========  ======================  ===========================

AutoDrive also takes *two consecutive* warped frames, not one.

``warped`` is produced with ``warpPerspective`` using a matrix ``C`` read
from ``config/homography_C_matrix.yaml``. That file is not committed to the
upstream repository, so the C++ pipeline throws during construction unless
you generate it. Upstream ships ``scripts/find_homography_C_matrix.py`` for
exactly that, because ``C`` depends on ``H``, which is camera specific.

``C`` is recoverable from two things upstream does commit: the hardcoded
matrix ``V`` (warped pixels to road plane) in ``inference.cpp``, and ``H``
(raw pixels to road plane). Since ``V · C = H``::

    C = V⁻¹ · H

Verified against upstream's own script: the two agree to 0.0035 px over a
grid of 1600 points, which is float32 versus float64 rounding.

World frame convention, from the comment at ``lateral_fusion.cpp:200``:
``x`` forward in metres, ``y`` lateral in metres, positive to the left.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# --- 70mai A800S intrinsics at 1920x1080 -----------------------------------
# From a chessboard calibration: 0.710 px training RMS, 0.649 px validation
# RMS over 30 images. The calibration reported "limited corner coverage", so
# the focal length has not been confirmed independently — see README.
CAMERA_MATRIX = np.array(
    [[1059.3747, 0.0, 967.2395], [0.0, 1059.0414, 568.9808], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DISTORTION = np.array([-0.38598, 0.16091, -5e-05, 6e-05, -0.03252], dtype=np.float64)
CALIBRATED_WIDTH, CALIBRATED_HEIGHT = 1920, 1080

# Focus of expansion: where the direction of travel projects onto the image.
# Measured from optical flow over 34 straight segments with outlier rejection;
# standard error 14 px (0.77 deg).
DEFAULT_FOE = (880.2, 654.2)

# --- constants copied from the Vision Pilot sources ------------------------
# modules/models/src/inference.cpp:100-103, marked "DO NOT MODIFY"
V_WARPED_TO_WORLD = np.array(
    [
        [0.00209514907, -0.000941721466, -9.24906396],
        [0.00662758637, -0.000352940531, -3.33396502],
        [0.000120077371, -0.00411343505, 1.0],
    ],
    dtype=np.float64,
)
# modules/models/src/inference.cpp:22-23
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NET_WIDTH, NET_HEIGHT = 1024, 512

# github-io/hardware.md calls the 52-55 deg horizontal field of view
# "a hard requirement, not a preference".
TARGET_HFOV_DEG = 53.0


def top_crop_2_1(height: int, width: int) -> int:
    """Port of ``compute_top_crop_2_1`` (``common/include/common/utils.hpp:15``)."""
    return max(0, int(round(height - width / 2.0)))


class VirtualCamera:
    """Turns a wide dashcam frame into Vision Pilot's two inputs.

    The dashcam sees 84.4 deg horizontally and is mounted by hand at an
    arbitrary angle. Vision Pilot expects roughly 53 deg, aligned with the
    direction of travel and pitched 1-3 deg down. A virtual camera with those
    properties is synthesised by rotating about the optical centre so the
    focus of expansion lands where the required mounting pose would put it.

    Undistortion, rotation and scaling collapse into a single remap table, so
    each output pixel is interpolated exactly once.
    """

    def __init__(
        self,
        width: int,
        height: int,
        foe: tuple[float, float] = DEFAULT_FOE,
        camera_height_m: float = 1.30,
        pitch_down_deg: float = 2.0,
        roll_deg: float = 0.0,
    ):
        if (width, height) != (CALIBRATED_WIDTH, CALIBRATED_HEIGHT):
            scale = width / CALIBRATED_WIDTH
            if abs(height / CALIBRATED_HEIGHT - scale) > 0.01:
                raise SystemExit(
                    f"\nSTOP: video is {width}x{height} but the calibration is for "
                    f"{CALIBRATED_WIDTH}x{CALIBRATED_HEIGHT}.\n"
                    "  A different aspect ratio means that recording mode CROPS the "
                    "frame\n  rather than scaling it, so the intrinsics do not "
                    "transfer. Calibrate\n  that mode separately.\n"
                )
            K = CAMERA_MATRIX.copy()
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
            foe = (foe[0] * scale, foe[1] * scale)
            print(
                f"  {width}x{height}: scaled K and FOE by {scale:.4f} from "
                f"{CALIBRATED_WIDTH}x{CALIBRATED_HEIGHT}"
            )
        else:
            K = CAMERA_MATRIX.copy()

        self.K, self.D = K, DISTORTION
        self.width, self.height = width, height
        self.camera_height_m = float(camera_height_m)
        self.roll_deg = float(roll_deg)

        focal = (NET_WIDTH / 2) / math.tan(math.radians(TARGET_HFOV_DEG / 2))
        self.K_virtual = np.array(
            [[focal, 0, NET_WIDTH / 2], [0, focal, NET_HEIGHT / 2], [0, 0, 1.0]]
        )
        # Pitching down by theta puts the direction of travel this far above
        # the image centre.
        foe_row = NET_HEIGHT / 2 - focal * math.tan(math.radians(pitch_down_deg))

        # The FOE was measured on the DISTORTED image, so undistort it before
        # turning it into a ray. With a lens this wide, skipping that step
        # tilts the whole virtual camera.
        undistorted = cv2.undistortPoints(
            np.array([[list(foe)]], np.float64), K, DISTORTION, P=K
        )
        ray_measured = np.linalg.inv(K) @ np.array(
            [undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0]
        )
        ray_target = np.linalg.inv(self.K_virtual) @ np.array(
            [NET_WIDTH / 2, foe_row, 1.0]
        )
        ray_measured /= np.linalg.norm(ray_measured)
        ray_target /= np.linalg.norm(ray_target)
        axis = np.cross(ray_measured, ray_target)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.dot(ray_measured, ray_target))
        self.R = (
            cv2.Rodrigues(axis / sin_a * math.atan2(sin_a, cos_a))[0]
            if sin_a > 1e-9
            else np.eye(3)
        )
        self.rotation_deg = math.degrees(math.atan2(sin_a, cos_a))

        self.map_resized = cv2.initUndistortRectifyMap(
            K, DISTORTION, self.R, self.K_virtual, (NET_WIDTH, NET_HEIGHT), cv2.CV_16SC2
        )

        # H for the virtual camera: pixels -> road plane (x fwd, y left, metres)
        forward = np.linalg.inv(self.K_virtual) @ np.array(
            [NET_WIDTH / 2, foe_row, 1.0]
        )
        forward /= np.linalg.norm(forward)
        image_x = np.array(
            [math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg)), 0.0]
        )
        right = image_x - np.dot(image_x, forward) * forward
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        world_to_pixel = self.K_virtual @ np.column_stack(
            [forward, -right, self.camera_height_m * down]
        )
        self.H = np.linalg.inv(world_to_pixel)

        self.C = np.linalg.inv(V_WARPED_TO_WORLD) @ self.H
        self.C /= self.C[2, 2]
        self.map_warped_x, self.map_warped_y, self.sensor_coverage = self._warp_table()

    def _warp_table(self):
        """Fold C and lens distortion into one remap: warped px -> source px."""
        C_inv = np.linalg.inv(self.C)
        cols, rows = np.meshgrid(
            np.arange(NET_WIDTH, dtype=np.float64),
            np.arange(NET_HEIGHT, dtype=np.float64),
        )
        homogeneous = np.stack([cols.ravel(), rows.ravel(), np.ones(cols.size)])
        virtual = C_inv @ homogeneous
        valid = np.abs(virtual[2]) > 1e-12
        virtual = virtual / np.where(valid, virtual[2], 1.0)

        rays = self.R.T @ (np.linalg.inv(self.K_virtual) @ virtual)
        valid &= rays[2] > 1e-9
        normalised = (rays / np.where(rays[2] > 1e-9, rays[2], 1.0)).T.reshape(-1, 1, 3)
        pixels = cv2.projectPoints(
            normalised, np.zeros(3), np.zeros(3), self.K, self.D
        )[0].reshape(-1, 2)
        valid &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= self.width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= self.height - 1)
        )
        return (
            pixels[:, 0].reshape(NET_HEIGHT, NET_WIDTH).astype(np.float32),
            pixels[:, 1].reshape(NET_HEIGHT, NET_WIDTH).astype(np.float32),
            valid.reshape(NET_HEIGHT, NET_WIDTH),
        )

    def prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(warped, resized)`` for one source frame."""
        warped = cv2.remap(
            frame,
            self.map_warped_x,
            self.map_warped_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        resized = cv2.remap(frame, *self.map_resized, cv2.INTER_LINEAR)
        return warped, resized

    @staticmethod
    def chw_imagenet(bgr: np.ndarray) -> np.ndarray:
        """Port of ``chw_imagenet`` (``inference.cpp:25-40``)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])

    @staticmethod
    def chw_01(bgr: np.ndarray) -> np.ndarray:
        """Port of ``chw_01`` (``inference.cpp:42-54``)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def describe(self) -> str:
        hfov = 2 * math.degrees(math.atan((self.width / 2) / self.K[0, 0]))
        return (
            f"  source {self.width}x{self.height} HFOV {hfov:.1f} deg -> "
            f"{NET_WIDTH}x{NET_HEIGHT} HFOV {TARGET_HFOV_DEG:.0f} deg\n"
            f"  virtual camera rotated {self.rotation_deg:.1f} deg onto the FOE, "
            f"pitched 2.0 deg down\n"
            f"  warped: C = V^-1 @ H(height={self.camera_height_m:.2f} m), "
            f"{100 * self.sensor_coverage.mean():.1f}% of pixels come from the sensor"
        )
