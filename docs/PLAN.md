# Plan — Measure It (Segmentation and Metric Measurement)

xis.ai AI/CV technical assessment · branch `dev-test01` · started 2026-11-05

**Machine:** Dell Latitude 5490 — Intel i5-8365U (4C/8T, 15 W), 15.8 GB RAM, Intel UHD 620, no CUDA, Windows 11.
**Camera:** phone main camera, focus/zoom/ISO/white balance locked, 4:3. Working resolution **2048×1536**; native
originals archived locally, the 2048×1536 set is the canonical input for calibration and measurement.
**Object:** flat rigid matte plate, about 120 × 80 × 4 mm, on the same base as the reference.
**Reference:** printed ArUco `DICT_4X4_50` marker (~40 mm black square) on card of matched thickness, measured
with a calliper — not trusted from the print file.

## 1. Order of work

| # | Work | Time | Deliverable |
|---|---|---|---|
| 0 | Repo, plan, objectives, DoD, frozen split manifest | 0:40 | this file + `dataset/splits.csv` |
| 1 | ChArUco board (A4, 9×7 squares, 25 mm), 30 views, calibration, saved model | 1:00 | `calibration/params/camera.json` + RMS report |
| 2 | Capture 100 dataset images + 12 accuracy instances + 25 tilt frames; undistort; label | 2:15 | dataset, masks, capture log |
| 3 | Train U-Net (MobileNetV3-Small encoder), select on validation only | 0:30 + clock | `models/runs/<id>/` |
| 4 | Test split: IoU, mAP@0.5, mAP@0.5:0.95, precision, recall | 0:20 | metrics + per-image table |
| 5 | One command, image → mm, explicit failure codes | 1:00 | `inference/measure.py` |
| 6 | 12-instance calliper study, raw vs undistorted | 0:40 | accuracy tables |
| 7 | Latency target, 5 runs, median and spread | 0:15 | timing report |
| 8 | Tilt sweep 0–40°, error vs angle | 0:40 | condition study |
| 9 | Refusal thresholds and their cost | 0:40 | threshold table |
| 10 | Docs filled with evidence, README, PR, 5-minute recording | 1:00 | three docs, PR link |

Total ≈ 8 h 05 m of work plus one background training run. Labeling is the schedule risk, not training.

## 2. Approach taken, and what was rejected before coding (item 10)

**Taken.** U-Net with a MobileNetV3-Small ImageNet encoder (`segmentation-models-pytorch`), 512² input,
BCE + Dice. Inference is coarse 512² → object box → second 512² pass on the full-resolution crop, because a
single 512² mask upsampled to 2048 px would put ~1 mm of boundary error into the measurement. Pixels become
millimetres through a homography from the four ArUco corners to a metric square; the object mask is warped into
that plane and measured as an oriented minimum-area rectangle. Refusal uses mask confidence, ±1 px boundary
stability, marker reprojection error, marker–object separation and plane tilt from `solvePnP`.

**Rejected.** Mask R-CNN / DeepLabV3+ ResNet (2–4 s per image here, unstable with ~70 images); SAM or SAM2 even
as a labeller (600 MB+, seconds per CPU image, and not my trained model); threshold/GrabCut as the final
segmenter (not trained, fails when lighting changes); Ultralytics YOLO and Roboflow (prohibited); ONNX Runtime
(no Python 3.15 wheel on this machine).

**Cost of rejecting.** Roughly 2× the latency of an ONNX export, about 25 extra minutes of labelling that SAM
would have saved, and no large-model prior — paid for with a pipeline I can explain line by line and that runs in
about one second on this laptop.

## 3. Dataset pre-registration (frozen before capture)

100 images, 10 sessions × 10 shots, split **by session** so near-duplicates cannot cross splits:
**S01–S07 train (70) · S08–S09 val (20) · S10 test (10)**. Distances 350–520 mm, rotations 0–45°, three
backgrounds (light grey, white paper, dark cloth) and three light sources (daylight, window, warm lamp) all occur
in training. The test session uses only conditions already seen in training, but a session never seen before.

`dataset/splits.csv` is the authority. Filenames are fixed there and captures are renamed to match; a bad frame
is excluded and declared, never replaced after seeing results. Two further sets are held out and pre-registered
separately — 12 calliper instances (`measurement/instance_manifest.csv`) and a 25-frame tilt sweep
(`measurement/conditions/tilt_manifest.csv`) — and are never used for training, tuning or checkpoint selection.

## 4. Deliberately not attempting

Multi-object scenes, instance segmentation, sub-pixel edge refinement beyond the crop pass, lighting as the
breaking condition (tilt is chosen), ONNX/quantisation, and any web UI. If time runs short I drop items 8 → 7 → 9
in that order and record it in the Definition of Done.

## 5. Plan changes (append-only; a plan that changed with a reason is better than one rewritten at the end)

| Date | Change | Why |
|---|---|---|
| 2026-11-05 | Initial plan committed before any code | — |
