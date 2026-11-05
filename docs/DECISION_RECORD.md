# Decision record — Measure It

**Branch:** `dev-test01` · **Date:** 2026-11-05
**Status:** Phase 4A software complete; all physical validation is still pending.

This record documents decisions that are implemented in the repository. It is
not an accuracy report and does not claim that any physical experiment has been
performed. Every number below is either a software constant or a count taken
from the committed files.

---

## 1. Evidence status

| Item | Status | Evidence |
|---|---|---|
| Camera calibration | **not performed** | `calibration/params/camera.json` does not exist |
| Physical reference marker size | **not measured** | `measurement/reference.json` has `marker_size_mm: null` |
| Dataset images | **0 / 100** | `dataset/report/dataset_report.md` |
| Masks / polygons | **0** | `dataset/report/dataset_report.md` |
| Trained checkpoint | **none** | `runs/` is local and gitignored |
| Accuracy vs instrument | **not measured** | no ground-truth CSV exists |
| Tilt study | **not measured** | no measured angles or tilt results exist |
| Real-artifact benchmark | **not run** | `evaluation.timing` exists; `results/` is gitignored |
| Refusal evidence | **synthetic software tests only** | `tests/test_evaluation_refusal_analysis.py` |

The Phase 4A test suite uses generated markers, synthetic cameras and temporary
fixtures. Those tests verify software behaviour; they are not physical evidence.

---

## 2. Measurement approach

**Decision:** measure one flat object with one planar ArUco reference marker.

1. The camera is calibrated once with a printed ChArUco board.
2. Each frame is undistorted with that calibration.
3. The configured ArUco marker is detected and validated: exactly one instance,
   convex, fully inside the frame, large enough, and not excessively
   foreshortened.
4. The marker's four corners are mapped to a square of the measured physical
   size. That homography supplies the metric scale and rectifies the plane.
5. The object mask is transformed into the rectified millimetre frame and
   reduced to a minimum-area oriented rectangle.
6. Width, height, angle and a rule-based reliability grade are reported.

**Assumption:** the marker and object lie on the same plane. A single view
cannot verify coplanarity; every successful result states the assumption in
`coplanarity_assumption` and `warnings`.

**Alternatives rejected:** multiple markers or a checkerboard in every frame
(more hardware and capture burden), a learned depth/pose model (no data and too
slow on CPU), and emitting a dimension whenever a mask exists (hides rejection
and scale failures).

---

## 3. Calibration assumptions

- Pinhole camera with OpenCV radial-tangential distortion
  (`opencv-radtan-4/5/8/12/14`).
- The ChArUco square size is **measured with the recorded instrument**, not
  assumed from the print file.
- Calibration and inference use the same working resolution. This project's
  target is **2048×1536**. A size mismatch is refused with
  `calibration_image_size_mismatch`.
- Undistortion happens before segmentation and marker detection.
- The calibration file is the only source of intrinsics. No camera matrix or
  distortion vector is hard-coded in the measurement path.
- The committed `camera.json` does not exist; a failed or absent calibration can
  never produce a millimetre value.

---

## 4. Segmentation architecture

**Decision:** U-Net with a MobileNetV3-Small encoder, trained on CPU.

| Item | Decision |
|---|---|
| Encoder | MobileNetV3-Small, ImageNet weights during training only |
| Decoder | five-level U-Net decoder from scratch |
| Input | RGB, 512×512, ImageNet mean/std |
| Loss | `0.5 × BCEWithLogits + 0.5 × (1 − soft Dice)` |
| Optimiser | AdamW, `ReduceLROnPlateau` |
| Selection | early stopping on validation IoU; `best.pt` keeps best validation IoU |
| Refinement | coarse 512² pass, then a full-resolution crop pass |
| Data | only genuine image+mask pairs from the frozen split |
| Determinism | fixed Python/NumPy/torch seeds and a pure augmentation function |

The model has **1,739,505 parameters**. No checkpoint is committed and no
training run has been performed.

**Rejected:** Mask R-CNN / DeepLabV3+ ResNet (too slow and data-hungry here),
SAM/SAM2 as the segmenter (large and slow on CPU, and not the trained model),
Ultralytics YOLO and Roboflow (prohibited by the brief), and ONNX Runtime (no
Python 3.15 wheel on this machine).

---

## 5. Thresholds and refusal rules

Defaults come from `measurement/reference.json` and are enforced in
`measurement/engine.py`:

| Setting | Default | Meaning |
|---|---|---|
| `dictionary` | `DICT_4X4_50` | ArUco dictionary |
| `marker_id` | `0` | exactly one reference marker |
| `marker_size_mm` | `null` | measured outer black square; inference refuses while null |
| `min_marker_px` | `60` | shortest marker side in pixels |
| `max_side_ratio` | `3.0` | marker foreshortening limit |
| `min_object_area_pct` | `0.2` | minimum mask area |
| `min_dimension_mm` | `0.5` | minimum measured dimension |
| `max_dimension_mm` | `2000.0` | maximum measured dimension |
| `border_margin_px` | `2` | marker/object border clearance |

A mask component is "significant" when its area is at least **5 %** of the
largest component. More than one significant component, or a mask touching the
frame border, is `partial_occlusion`.

`reliability` is a rule-based geometry grade, **not** a learned probability:

- **high:** marker at least 2× `min_marker_px`, side ratio ≤ 1.5, object area at
  least 1 % and 2× `min_object_area_pct`, and generous border clearance.
- **medium:** marker at least 1.25× `min_marker_px` and side ratio ≤ 2.5.
- **low:** otherwise.

### Refusal and exit codes

| Outcome | Code | Exit |
|---|---|---|
| measurement produced | — | 0 |
| usage error / unreadable image / unwritable output | — | 2 |
| calibration missing, invalid or wrong size | `missing_calibration`, `invalid_calibration`, `calibration_image_size_mismatch` | 3 |
| reference config missing/invalid, or size null | `missing_reference_config`, `invalid_reference_config`, `reference_size_unknown` | 3 |
| checkpoint missing/invalid | `missing_checkpoint`, `invalid_checkpoint` | 3 |
| no reference marker | `missing_reference` | 4 |
| no object mask | `missing_object` | 4 |
| split mask / border contact | `partial_occlusion` | 4 |
| degenerate geometry or implausible size | `invalid_geometry` | 4 |

A failed result still writes structured JSON with `measurement: null` or a
rejection record. It never emits a millimetre value from missing physical
artifacts.

**Excessive tilt is not implemented as a refusal rule.** The engine limits
foreshortening through `max_side_ratio` but does not compute a plane angle or
refuse at a configured tilt angle. `evaluation.refusal_analysis` reports that
gap explicitly. A tilt threshold must be added only after real tilt data exists
and the threshold is justified; it must not be invented in advance.

---

## 6. Evaluation and reporting decisions

| Tool | Decision |
|---|---|
| `evaluation.raw_vs_undistorted` | measures raw and undistorted frames independently; difference is `undistorted − raw` |
| `evaluation.condition_study` | groups by a **measured** tilt angle; never substitutes `target_angle_deg` |
| `evaluation.timing` | times model load, undistort, segmentation, marker detection, geometry and total; reports median/p95 and labels itself software-only |
| `evaluation.accuracy_study` | requires manual ground truth with a named instrument; reports MAE, RMSE, mean percentage error, maximum absolute error and failure/refusal rate |
| `evaluation.refusal_analysis` | checks observed refusal codes and categories against expected ones, and reports uncovered categories |

Error definitions:

- **MAE:** mean of `|system − ground truth|` in millimetres.
- **RMSE:** `sqrt(mean((system − ground truth)²))` in millimetres.
- **Mean percentage error:** mean of `100 × |system − ground truth| / ground truth`.
- **Maximum absolute error:** largest `|system − ground truth|`.
- **Failure / refusal rate:** instances without a valid measurement divided by
  all ground-truth instances.

Accuracy metrics are computed on successful measurements only. The failure
rate is reported separately so a high refusal rate cannot be hidden behind a
good MAE.

**Decision:** synthetic fixtures are allowed only in unit tests and must be
labelled as synthetic in every refusal-analysis report. They are never treated
as real-world evidence and are never used for training.

---

## 7. Known limitations

- No camera has been calibrated, no dataset image exists, no model has been
  trained and no measurement has been compared with an instrument.
- Marker/object coplanarity is assumed and cannot be verified from one view.
- One object, one class, one plane; no instance segmentation or multi-object
  scenes.
- Occlusion handling is limited to border contact and split components. A
  partially hidden object that still forms one component may be measured
  incorrectly.
- The reliability grade is a geometry heuristic, not a calibrated confidence.
- No excessive-tilt refusal threshold exists.
- The camera model is pinhole with radial-tangential distortion; unusual lenses
  or strong rolling-shutter effects are out of scope.
- Benchmark timings depend on the machine, thread count, torch build and power
  state. They are not portable performance guarantees.
- Ground truth is only as good as the person and instrument that produced it.

---

## 8. Trade-offs

| Trade-off | Choice | Consequence |
|---|---|---|
| Model capacity vs CPU latency | MobileNetV3-Small U-Net | fast enough for CPU, but less expressive than a large model |
| Mask resolution vs boundary error | 512² coarse + full-resolution crop | extra inference pass, better boundary than a single upsample |
| Single marker vs multi-marker | one planar marker | simple capture, but scale depends on one measured print and coplanarity |
| Refusal vs always answer | refuse when physical artifacts or geometry fail | fewer numbers, but every emitted number has a required basis |
| Synthetic tests vs real evidence | synthetic-only unit tests | fast and reproducible, but explicitly not accuracy evidence |
| ONNX vs PyTorch | PyTorch CPU | no Python 3.15 ONNX wheel here; slower than a tuned runtime |
| Tilt refusal vs no threshold | no tilt threshold yet | avoids inventing a limit; tilt risk remains open |

---

## 9. Remaining risks

1. **Calibration risk:** a poor ChArUco capture or wrong square size biases
   every measurement.
2. **Marker-size risk:** `marker_size_mm` is a physical measurement; a print or
   caliper error scales every result.
3. **Coplanarity risk:** if the object or marker is lifted, the homography is
   wrong even when the marker is detected cleanly.
4. **Segmentation risk:** domain shift in lighting, background or object pose
   can produce a plausible but wrong mask.
5. **Tilt risk:** no validated tilt threshold exists; large tilt may bias the
   measurement or cause refusals.
6. **Ground-truth risk:** operator error, instrument resolution or a different
   measurement face invalidates the accuracy study.
7. **Representativeness risk:** 100 images from 10 sessions may not cover the
   deployment population.
8. **Performance risk:** the benchmark is machine-specific; a different CPU or
   thread count changes latency.
9. **Dependency risk:** pinned library versions and the Python 3.15 environment
   must be reproduced for the recorded results to remain comparable.

---

## 10. Required before sign-off

- [ ] Print the ChArUco board, measure a square with the recorded instrument,
  and complete a passing calibration.
- [ ] Print the ArUco marker, measure its outer black square, and set
  `marker_size_mm`.
- [ ] Capture, ingest and annotate the frozen dataset.
- [ ] Train and evaluate the segmentation model on the frozen split.
- [ ] Measure A01–A12 with the recorded instrument and create the ground-truth
  CSV.
- [ ] Capture the measured tilt sweep; record `measured_angle_deg` for every
  frame.
- [ ] Run all Phase 4A reports on real artifacts.
- [ ] Decide whether a tilt threshold is justified by the real tilt data.
- [ ] Update `docs/OBJECTIVES.md` and `docs/DEFINITION_OF_DONE.md` with raw
  evidence; do not tick a box without a file or result.
- [ ] Record any target that was missed, with the reason.

---

## 11. Sign-off

| Role | Name | Date | Decision |
|---|---|---|---|
| Author |  |  |  |
| Reviewer |  |  |  |

Until the checklist above is complete, the correct status is **software
complete, physical validation pending**.
