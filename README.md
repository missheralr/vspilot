# visionpilot-dashcam

Run [Autoware Vision Pilot](https://github.com/autowarefoundation/autoware_vision_pilot)'s
perception models on footage from a consumer dashcam — a 70mai A800S — without
building the C++ stack.

A wide-angle dashcam is not the camera Vision Pilot expects. This project
bridges the gap: it synthesises a virtual camera with the required field of
view and mounting pose, feeds the three ONNX models the two differently
normalised inputs they actually want, tracks the results over time, and
renders an annotated video with a metrics panel.

![panel](docs/panel.jpg)

---

## What it does

| Stage | Detail |
|---|---|
| Lens correction | Calibrated intrinsics, single remap table, one interpolation per output pixel |
| Virtual camera | 84.4° source rotated onto the focus of expansion, re-rendered at 53° |
| Model inputs | `warped` (ImageNet normalised) and `resized` (÷255), built as upstream does |
| Inference | AutoDrive, AutoSteer, AutoSpeed, run concurrently |
| Ego speed | Decoded from the GPS log embedded in the MP4 — no ffmpeg needed |
| Tracking | Constant-velocity Kalman filters for distance, cross-track error, heading, curvature |
| Planning | IDM longitudinal planner, ported from upstream |
| Output | Annotated MP4 plus a per-frame CSV |

---

## Quick start

```bash
pip install -r requirements.txt
python download_models.py          # once, 114 MB
python main.py clip.mp4 --out results --hz 10 --camera-height 1.32
```

Results land in `results/annotated.mp4` and `results/metrics.csv`.

On Windows, double-click **`RUN.bat`** instead: it checks for Python, installs
the dependencies on first use and opens a small window with a file picker. No
terminal required.

### Useful options

| Option | Meaning |
|---|---|
| `--hz 5` | Halve the work; Vision Pilot targets 10 Hz |
| `--from-s 40 --to-s 60` | Process one span only |
| `--camera-height 1.32` | Height of the camera above the road, metres |
| `--ego-speed-kmh 30` | Force a speed when the video has no GPS log |
| `--headway 1.0` | IDM desired time headway — see the note below |
| `--no-tracking` | Treat each frame independently, for comparison |

---

## How the preprocessing was worked out

Vision Pilot does not take one image per frame. Reading
`modules/models/src/inference.cpp` and
`modules/sensing/image_preprocessing/src/image_preprocessor.cpp`:

| Image | Consumed by | Normalisation |
|---|---|---|
| `warped` | AutoDrive | ImageNet mean/std |
| `resized` | AutoSteer, AutoSpeed | divide by 255 only |

AutoDrive also takes **two consecutive** warped frames, not one.

`warped` is produced by `warpPerspective` with a matrix `C` loaded from
`config/homography_C_matrix.yaml`. That file is **not committed upstream**, so
the C++ pipeline throws during construction until you generate it — upstream
ships `scripts/find_homography_C_matrix.py` for the purpose, because `C`
depends on `H`, which is camera-specific.

`C` follows from two things upstream does commit: the hardcoded matrix `V`
(warped pixels → road plane) and `H` (raw pixels → road plane). Since
`V · C = H`:

```
C = V⁻¹ · H
```

Checked against upstream's own script: the two agree to **0.0035 px** across a
grid of 1600 points, which is float32 against float64 rounding.

`H` for this dashcam is built from the intrinsics, the measured focus of
expansion and the camera height.

---

## Temporal tracking

Without it, every frame is independent: the numbers jitter and there is no way
to tell whether the gap is closing. Upstream uses particle filters
(`safety_guardian/fusion`, ~1100 lines); this uses constant-velocity Kalman
filters to the same end, and they still produce the standard deviation that
upstream exposes as `distance_stddev_m`.

Measured on a real 180 s clip, mean absolute change per frame:

| Quantity | Untracked | Tracked | Reduction |
|---|---|---|---|
| Lead distance | 2.40 m | 0.54 m | **78 %** |
| Cross-track error | 0.36 m | 0.11 m | 69 % |
| Heading error | 8.15° | 4.44° | 45 % |

Two independent distance measurements are fused, as `LongitudinalFusion` does:
AutoDrive's own estimate, and the distance implied by projecting a detection's
box bottom onto the road plane.

---

## Performance

The three models run concurrently, as `inference.cpp` does with `std::async`.
Running them sequentially wastes most of the available speed-up, because
onnxruntime releases the GIL during a session.

| | Median per frame |
|---|---|
| Sequential | 389 ms |
| Concurrent | **209 ms** (1.86×) |

That is on a 2-core machine; more cores widen the gap. Outputs are identical
either way — the self-test asserts it.

Vision Pilot targets 10 Hz. Reaching that from here, in order of effect:

1. **GPU.** The only reliable route. Upstream ships `trt_engine` for TensorRT.
2. **More CPU cores.** Three models in parallel means 6–8 cores gets close.
3. **int8 — only if the CPU has VNNI.** Measured on a CPU without it, int8 was
   *slower* than fp32 (1409 ms against 950 ms) because the kernels are
   emulated. Check before assuming.
4. **Drop to 5 Hz.** Tracking still works, just coarser.

Running over files does not need real time. A live RTSP feed would.

---

## IDM and dense urban traffic

Over a full 180 s clip the planner asked for "hard brake" in 21 % of frames.
That is not a defect — the default parameters do not match the environment:

| | |
|---|---|
| Measured time headway on the clip | **1.09 s** (median) |
| IDM default expectation | **1.50 s** + 2 m |
| Frames below 1.5 s headway | **66 %** |

The 1.5 s default comes from upstream and suits Western highway driving.
Re-running the same clip:

| Parameters | Hard brake | Cruise |
|---|---|---|
| `T=1.5 s, s0=2.0 m` (default) | 12 % | 54 % |
| `T=1.0 s, s0=1.5 m` | 0 % | 73 % |
| `T=0.6 s, s0=1.0 m` | 0 % | 98 % |

Defaults stay faithful to upstream. Pass `--headway 1.0 --min-gap 1.5` for
something closer to the traffic this footage was shot in.

Note that IDM is unbounded below, and so is upstream — it returns the raw
value. In a near-collision geometry it can demand about −146 m/s². That is the
algorithm, not a bug.

---

## How far to trust each output

| Output | Status |
|---|---|
| Ego path band | Trustworthy; verify by eye against the video |
| Detection boxes | Trustworthy; verify by eye |
| Objects in ego corridor | Trustworthy — this is the count worth reading, since the raw count includes roadside clutter |
| Cross-track error, heading, curvature | Computed exactly as `lateral_fusion` does, but never compared against ground truth |
| Ego speed | Trustworthy; the GPS decode cross-checks to 2.1 % |
| **Distances in metres, TTC, acceleration** | **Not validated** — see below |

Lead distance was cross-checked two independent ways: AutoDrive's own output
from the `warped` image, against the distance implied by the detection box
bottom. A linear fit gave a slope of **0.57** where it should be 1, with an
RMS of 8 m against a median distance of 9.4 m. Four explanations remain open
and this project has not separated them: a different vehicle being tracked by
each path, an error in the warped geometry, box bottoms not sitting at the
ground contact point, or `150 · (1 − x)` not matching this checkpoint.

### Parameters that are measured, assumed, or unverified

| Parameter | Status |
|---|---|
| Intrinsics, distortion | Measured. 0.710 px training RMS, 0.649 px validation, 30 images. The calibration warned of limited corner coverage. |
| Focus of expansion | Measured from optical flow, 34 straight segments, standard error 14 px (0.77°) |
| Camera height | **Assumed 1.30 m.** A 0.2 m error moves the AutoDrive distance by up to 17 m (>50 %). Measure it. |
| Focal length | **Never independently verified.** Straight-line constraints cannot separate focal length from radial distortion — tested, and *f* from 350 to 1400 produced identical residuals (RMS 4.226 px). To settle it: photograph a 300 mm board at exactly 2.00 m; *f*=1059 predicts 159 px. |
| Roll | **Assumed zero.** Not measured. |

---

## When this does not apply

The calibration and focus of expansion belong to one specific 70mai A800S in
one specific mounting position, recording 1920×1080.

Still fine: other clips from the same camera in the same position; a
proportionally scaled recording (handled automatically).

Not fine: a different camera; a camera that was removed and refitted or
knocked; a recording mode that *crops* rather than scales (detected, and the
program stops with an explanation).

A sign the camera has shifted: the path band sits off to one side in **every**
frame, rather than varying with the scene.

---

## Scope

This ports Vision Pilot's **perception** layer and a slice of its planning. It
is not the whole system. Not ported: the particle filters themselves, radar
association, the MPC lateral planner, and any CAN output. Nothing here drives
a vehicle.

---

## Layout

```
main.py             command-line entry point
gui.py              desktop window, no terminal needed
RUN.bat             Windows launcher
calibration.py      camera model, virtual camera, the two model inputs
models.py           ONNX sessions, concurrent inference, output decoding
geometry.py         waypoints and detections projected onto the road plane
tracking.py         Kalman filters, scene tracker, IDM planner
render.py           path band and information panel
gps_log.py          MP4 GPS log reader, pure Python
download_models.py  fetch the weights
self_test.py        every check in one run
```

Run the checks:

```bash
python self_test.py clip.mp4
```

39 checks, including the geometric round trip, the IDM against a hand
calculation, the Kalman gate, parallel-equals-serial, and panel overflow
across all 16 state combinations.

---

## Licence

Apache 2.0. See `LICENSE` and `NOTICE` — this derives from
`autowarefoundation/autoware_vision_pilot`, also Apache 2.0, and `NOTICE`
lists exactly what was ported. Model weights are downloaded from that
repository at runtime and are not redistributed here.
