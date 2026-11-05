# Phase 4B — Physical Readiness Audit

**Branch:** `dev-test01` · **Date:** 2026-11-06

This document records the actual physical state of the project. It was produced by
inspecting the repository and its data directories. **Nothing was created,
fabricated, calibrated, trained, measured or invented during this audit.**
Synthetic test fixtures are explicitly excluded from all evidence statements.

## 1. Audit method

The following locations were inspected:

- `calibration/params/camera.json`
- `calibration/board/` and `calibration/board/board_spec.json`
- `calibration/report/`
- `dataset/images/`, `dataset/masks/`, `dataset/annotations/`
- `dataset/capture_log.csv`, `dataset/report/dataset_report.md`
- `measurement/reference.json`
- `measurement/conditions/`
- `runs/`, `results/`
- `measurement/ground_truth*.csv`
- frozen split: `dataset/splits.csv` and its SHA-256

No files were modified by the audit except this document.

## 2. Physical readiness audit

| # | Requirement | Status | Evidence / path | Required action | Blocks |
|---|---|---|---|---|---|
| 1 | Printed ChArUco calibration board | **Missing** | Only digital board files exist: `calibration/board/charuco_board.pdf`, `charuco_board.png`, `board_spec.json` | Print the committed board at 100 % scale and keep the print flat | Calibration |
| 2 | Measured ChArUco square size | **Missing** | No measured square size recorded in `board_spec.json` | Measure one square with the real measuring instrument and record it with `python -m calibration.make_board measure` | Calibration |
| 3 | At least 30 genuine calibration photographs | **Missing** | `dataset/images/` does not exist; dataset report says 0 / 100 images | Capture and ingest 30+ board views at 2048×1536 | Calibration |
| 4 | Printed ArUco reference marker | **Missing** | Only software configuration exists: `measurement/reference.json` and `measurement/reference.py` | Print ArUco `DICT_4X4_50`, marker ID `0`, at a size that can be measured reliably | Measurement validation |
| 5 | Measured outer black marker size (mm) | **Missing** | `measurement/reference.json` has `marker_size_mm: null` | Measure the printed marker's outer black square with the real instrument and set `marker_size_mm` | Measurement validation |
| 6 | 100 genuine object photographs across 10 sessions | **Missing** | `dataset/images/` does not exist; dataset report says 0 / 100 | Capture 100 images according to the frozen split and session plan | Training and measurement validation |
| 7 | Validated masks and polygon annotations | **Missing** | `dataset/masks/` and `dataset/annotations/` do not exist; 0 annotations | Annotate all 100 images; validate mask/polygon agreement at IoU ≥ 0.98 | Training |
| 8 | Measuring instrument with documented accuracy | **Unknown / not documented** | No `capture_log.csv`, ground-truth CSV or instrument record exists | Identify the instrument, record its resolution/accuracy and operator in every manual measurement | Accuracy validation |
| 9 | At least 12 manually measured object instances | **Missing** | No `measurement/ground_truth*.csv` exists | Measure A01–A12 (or equivalent held-out instances) with the real instrument | Accuracy validation |
| 10 | At least 25 tilt-condition photographs with measured angles | **Missing** | `measurement/conditions/tilt_manifest.csv` contains only target angles; no measured angles or tilt images | Capture 25+ tilt frames and record `measured_angle_deg` for each | Tilt analysis |

## 3. Available software and non-physical assets

These are genuine repository assets, but they are not physical evidence:

- Frozen split manifest: `dataset/splits.csv`, SHA-256 verified and unchanged.
- ChArUco board generator and committed board files:
  - `calibration/board/charuco_board.pdf`
  - `calibration/board/charuco_board.png`
  - `calibration/board/board_spec.json`
- Dataset ingestion, quality, annotation and reporting software.
- U-Net segmentation software, frozen-split loader, metrics and CPU trainer.
- Measurement engine, ArUco reference validation, homography geometry and inference CLI.
- Phase 4A evaluation and reporting tools.
- 188 passing software tests, all using temporary synthetic fixtures.

No real image, mask, checkpoint, calibration file, marker size, ground-truth file or tilt result exists.

## 4. Commands verified

The following commands were checked with `--help` during this audit and returned a usable usage message:

- `python -m calibration.make_board`
- `python -m calibration.calibrate`
- `python -m dataset.ingest`
- `python -m dataset.annotate`
- `python -m dataset.reporting`
- `python -m segmentation.train`
- `python -m segmentation.evaluate`
- `python -m measurement.infer`
- `python -m evaluation.raw_vs_undistorted`
- `python -m evaluation.condition_study`
- `python -m evaluation.timing`
- `python -m evaluation.accuracy_study`
- `python -m evaluation.refusal_analysis`

`dataset.quality` is a library module used by ingestion; it is not a standalone CLI and was not expected to provide `--help`.

## 5. Safe checks actually performed

- Full software test suite: **188 passed**.
- Frozen split SHA-256: **unchanged**.
- Filesystem audit of all calibration, dataset, measurement, run and result paths.
- No calibration, training, inference, benchmark or accuracy command was run against fabricated inputs.

No physical experiment was completed.

## 6. Blocked experiments

All physical experiments are blocked by missing inputs:

- **Camera calibration:** blocked by missing printed board, measured square size and calibration photographs.
- **Dataset validation:** blocked by missing photographs and masks.
- **Model training/evaluation:** blocked by missing real image/mask pairs.
- **Measurement validation:** blocked by missing calibration, printed/measured marker and object photographs.
- **Accuracy validation:** blocked by missing instrument and 12 manual measurements.
- **Tilt analysis:** blocked by missing tilt photographs and measured angles.
- **Raw-vs-undistorted, benchmark, refusal analysis:** software is ready, but no real artifact exists to run them against.

## 7. Capture and measurement instructions for the developer

The developer uses a **Google Pixel 8 Pro**. Do not assume a vernier calliper is available. Use whatever measuring instrument is available, but record its type, resolution, accuracy and operator with every measurement.

### 7.1 ChArUco calibration board

1. Print `calibration/board/charuco_board.pdf` at **100 % scale**. Do not use
   “fit to page”, “shrink to fit” or browser scaling.
2. Verify the print is flat and undamaged.
3. Measure one square side with the real measuring instrument.
4. Record the measured square size:
   ```powershell
   python -m calibration.make_board measure
   ```
   Use `python -m calibration.make_board --help` and the subcommand help for the
   exact arguments on this machine.
5. Check the resulting `board_spec.json` and keep the measured value with the
   board.

### 7.2 Calibration photographs

Capture at least 30 views with the Pixel 8 Pro:

- Working resolution: **2048×1536** (4:3).
- Use the main wide lens at 1.0× zoom.
- Lock focus, ISO and white balance where the camera app allows.
- Keep the board flat and fully visible.
- Vary angle, distance and position; avoid blur, glare and motion.
- Include views near the frame edges and corners.
- Do not crop or resize manually; ingestion normalises resolution.

Then run:
```powershell
python -m calibration.calibrate `
    --images <calibration_image_dir> `
    --board calibration/board/board_spec.json `
    --out calibration/params/camera.json `
    --report-dir calibration/report `
    --working-size 2048x1536
```

Inspect `calibration/report/` and the RMS before trusting the calibration.
Do not edit `camera.json` by hand.

### 7.3 ArUco reference marker

1. Print ArUco `DICT_4X4_50`, marker ID `0`.
2. Choose a size that is practical to measure and large enough in the frame
   (about 30–50 mm outer black square is reasonable).
3. Measure the **outer black square side** with the real instrument.
4. Record the value and instrument, then update `measurement/reference.json`:
   ```json
   "marker_size_mm": <measured value>
   ```
5. Mount the marker and object on the same flat base so they are coplanar.
   Match thicknesses where possible.

### 7.4 Dataset capture (100 images, 10 sessions)

Follow `docs/PLAN.md` and the frozen split in `dataset/splits.csv`:

- 10 sessions, 10 images per session.
- Object spans **25–30 % of frame width**.
- Same base plane for marker and object.
- Consistent lighting; avoid hard shadows and reflections.
- Grey or neutral background.
- Correct orientation; do not rotate after capture.
- Pixel 8 Pro: 2048×1536, main lens, 1.0× zoom, locked settings.

Ingest each session:
```powershell
python -m dataset.ingest ingest `
    --source <incoming_dir> `
    --session S01 `
    --condition <condition_label>
```

Then validate and report:
```powershell
python -m dataset.reporting
```

Do not alter `dataset/splits.csv` or its SHA-256.

### 7.5 Annotation

For every image:
- create a mask and, where used, a polygon;
- validate with `python -m dataset.annotate validate`;
- check mask/polygon agreement at IoU ≥ 0.98.

### 7.6 Tilt-condition capture

Capture at least 25 frames:
- measured tilt angles such as 0, 10, 20, 30 and 40 degrees;
- each angle measured with a real instrument (digital level, protractor or
  equivalent), not estimated from the target;
- marker and object remain coplanar;
- record `measured_angle_deg`, `target_angle_deg`, condition and notes.

Create a manifest such as `measurement/conditions/tilt_measurements.csv`:
```csv
stem,measured_angle_deg,target_angle_deg,condition,notes
T00_01,0.4,0,fixed_400mm_daylight,digital level
```

Run inference on every frame, then:
```powershell
python -m evaluation.condition_study `
    --manifest measurement/conditions/tilt_measurements.csv `
    --images dataset/images `
    --results results/tilt `
    --ground-truth measurement/ground_truth.csv `
    --out results/condition_study
```

Do not substitute a target angle for a measurement.

### 7.7 Manual accuracy measurements

For at least 12 held-out instances:
- measure width and height with the real instrument;
- take at least three readings per dimension and use the mean;
- record instrument name, resolution/accuracy, date and operator;
- create `measurement/ground_truth.csv`:
```csv
stem,width_mm,height_mm,instrument,measured_utc,notes
A01,120.2,80.1,digital calliper,2026-11-05,three readings
```

Then run inference on those instances and:
```powershell
python -m evaluation.accuracy_study `
    --ground-truth measurement/ground_truth.csv `
    --results results/accuracy_instances `
    --out results/accuracy
```

The tool will not generate or estimate ground truth.

### 7.8 Other Phase 4A commands after real artifacts exist

Raw vs undistorted:
```powershell
python -m evaluation.raw_vs_undistorted `
    --image <real_image> `
    --camera calibration/params/camera.json `
    --reference measurement/reference.json `
    --checkpoint runs/seg/best.pt `
    --out results/raw_vs_undistorted
```

Benchmark:
```powershell
python -m evaluation.timing `
    --image <real_image> `
    --camera calibration/params/camera.json `
    --reference measurement/reference.json `
    --checkpoint runs/seg/best.pt `
    --out results/benchmark
```

Refusal analysis:
```powershell
python -m evaluation.refusal_analysis `
    --cases measurement/conditions/refusal_cases.csv `
    --images dataset/images `
    --results results/refusals `
    --out results/refusal_analysis
```

### 7.9 What not to do

- Do not create a fake `calibration/params/camera.json`.
- Do not set an invented `marker_size_mm`.
- Do not generate fake dataset images, masks or annotations.
- Do not train on synthetic data.
- Do not invent accuracy, tilt or timing results.
- Do not tick an objective or definition-of-done box without a file or saved
  result as evidence.

## 8. Experiments actually completed

None. This phase is an audit and readiness document only. The 188 passing tests
are software tests and are not physical validation.

## 9. Experiments blocked by missing inputs

- Camera calibration: blocked.
- Dataset ingestion and validation: blocked.
- Segmentation training and evaluation: blocked.
- Measurement validation: blocked.
- Accuracy validation: blocked.
- Tilt analysis: blocked.
- Real-artifact raw-vs-undistorted, benchmark and refusal analysis: blocked.

## 10. Exact actions required from the developer

1. Identify and document the measuring instrument (type, resolution, accuracy).
2. Print the ChArUco board at 100 % scale and measure one square.
3. Capture and ingest 30+ calibration photographs.
4. Run calibration and inspect the report.
5. Print and measure the ArUco marker; set `marker_size_mm`.
6. Capture 100 real object photographs across 10 sessions.
7. Annotate and validate all masks.
8. Train on the real training split and evaluate once on the frozen test split.
9. Measure at least 12 held-out instances and create the ground-truth CSV.
10. Capture at least 25 tilt frames with measured angles.
11. Run all Phase 4A reports on real artifacts.
12. Update `docs/OBJECTIVES.md` and `docs/DEFINITION_OF_DONE.md` with evidence.

## 11. Files created or modified

- `docs/PHASE_4B_READINESS.md` (new)

No other files were modified during this audit. `dataset/splits.csv` and its
SHA-256 were not changed.

## 12. Test results

- Full software suite: **188 passed**.
- No physical tests were run because no physical inputs exist.

## 13. Commit hashes

- `70db1a0` docs: add Phase 4B physical readiness audit

## 14. Honest overall assessment

The software is ready for physical validation, but physical validation has not
started. Calibration, dataset capture, training, measurement validation, tilt
analysis and accuracy validation are all blocked by missing real inputs. The
project remains in a software-complete, physical-validation-pending state.

**STOP: developer review and physical data collection are required before any
further experiment.**