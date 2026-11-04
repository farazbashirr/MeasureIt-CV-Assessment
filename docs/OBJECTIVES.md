# Objectives

Targets were fixed on 2026-11-05, before any photograph was taken and before any code existed. Every result
below is added later, copied from a saved script output; no number here is estimated or rounded in my favour.
**Status of all rows: pending.**

## MO-1 — Measurement accuracy (required)

| Field | Entry |
|---|---|
| What is measured | Absolute error of the reported width and height, in millimetres, against calliper ground truth, over 12 held-out instances (A01–A12) that were never used for training, validation or threshold tuning. |
| How | `evaluation/accuracy_study.py` compares each saved `measurement/results/A*.json` with `measurement/ground_truth.csv` (three calliper readings per dimension, mean used). It writes every instance, the mean absolute error, the mean percentage error, the median and the worst case. |
| Target | MAE ≤ **1.0 mm** and worst absolute error ≤ **2.0 mm** on both width and height; at least 10 of 12 instances within 1.5 mm on both dimensions. |
| Conditions | 2048×1536 working images, 380–450 mm working distance, plane tilt ≤ 15°, object and reference coplanar on the same matte base, locked phone camera, indoor daylight plus one lamp. |
| Not included | Occluded or partly hidden reference, tilt > 15°, frames rejected by the blur gate, objects outside the trained class, and dimensions taken on any face other than the top face. |

Why 1.0 mm: on a ~120 mm part this is 0.8 %, a sensible band for a non-critical fit check. At roughly 12 px/mm,
one pixel of boundary error on each side already costs ~0.17 mm, so 1 mm is reachable but not automatic. A target
that could be cleared without doing the work would prove nothing.

## MO-2 — End-to-end latency on CPU

| Field | Entry |
|---|---|
| What is measured | Wall-clock time of the single measurement command on a 2048×1536 image, from process start to JSON written, with the model already loaded. |
| How | `evaluation/timing.py`, five consecutive runs. Median plus min/max reported, with a per-stage breakdown (undistort, segment, reference, measure). Cold start (model load) reported separately and excluded from the median. |
| Target | Median ≤ **3.0 s**; spread (max − min) ≤ **1.0 s**. |
| Conditions | i5-8365U, 4 threads, AC power, Best performance plan, no other user load, `torch.set_num_threads(4)`. |
| Not included | The first run after boot, concurrent calls or builds, GPU execution (none available on this machine). |

## MO-3 — Robustness to plane tilt (the breaking condition, item 8)

| Field | Entry |
|---|---|
| What is measured | Width error in millimetres as the angle between the object plane and the sensor plane increases (0°, 10°, 20°, 30°, 40°). |
| How | `evaluation/condition_study.py` over the pre-registered 25-frame sweep (five instances per angle). The angle is not assumed: it is computed from `solvePnP` on the ArUco marker for every frame and logged next to the error. |
| Target | MAE ≤ **2.0 mm** up to 20°; and the refusal threshold must sit at the first angle where error exceeds **3.0 mm**, with that angle stated. |
| Conditions | Fixed 400 mm distance, same camera lock, same base and lighting, object and marker coplanar at 0°. |
| Not included | Tilt combined with distance change, tilt beyond 40°, and motion-blurred frames. |

Why tilt rather than distance or lighting: the pose gives an objective x-axis in degrees, it is the assumption the
whole homography depends on, and it feeds directly into the refusal logic rather than being a separate experiment.

## How numbers are reported

- Every figure comes from a file in `measurement/results/` or `models/runs/` and is pasted raw; the docs point at
  the file. Nothing is typed from memory.
- Timings: five runs, median and spread, never a single reading.
- Errors: every instance in a table, then mean, median and worst case. A mean alone hides the one measurement
  that was 15 mm out.
- Any target missed is reported as missed, with the reason and what it would have cost to fix.
