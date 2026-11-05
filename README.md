# Measure It — camera-based dimension measurement

An end-to-end computer-vision pipeline that measures a flat object in millimetres
from a single phone photograph. It is being built as the "Measure It" CV
assessment: own labelled dataset, own CPU model, one-command inference, and an
accuracy table checked against a real measuring instrument.

**Branch:** `dev-test01` · **Python:** 3.15 · **CPU only** (no CUDA, no YOLO,
no Roboflow model).

---

## Status — read this before trusting any number

| Area | State |
|---|---|
| Phase 0 — plan, objectives, definition of done, frozen split | complete (`docs/`) |
| Phase 1 — board generator, camera model, calibration pipeline | software complete, **physical calibration not performed** |
| Phase 2 — ingestion, quality checks, annotation tooling, reporting | software complete |
| Phase 3 — U-Net segmentation, measurement engine, inference CLI | **software complete; model not trained** |
| Phase 4A — experiments, benchmark, accuracy/refusal reporting | **software complete; no physical results generated** |
| Dataset images | **0 / 100** — no real photographs have been captured yet |
| Annotations | **0** — no masks or polygons exist |
| Segmentation model | implemented, **not trained**; no checkpoint exists |
| Measurement accuracy vs instrument | not measured |
| Tilt study | not measured |
| Real-artifact benchmark | not run |

`calibration/params/camera.json` and `calibration/report/` **do not exist**. They
are only written by a real passing calibration run, and no such run has
happened. Nothing in this repository fabricates images, masks, measurements or
metrics. `dataset/report/dataset_report.md` currently says `0 / 100`, which is
the honest state.

Phase 3 and 4A tests use generated markers, synthetic cameras and known
transformations. They verify the software's geometry, reporting and refusal
paths; they are **not** real-world measurement evidence and say nothing about
accuracy against an instrument. The decision record is
`docs/DECISION_RECORD.md`.

---

## How the pipeline is meant to work

1. **Calibrate the camera** with a printed ChArUco board → `camera.json`
   (intrinsics + distortion). The board's square size is measured with a real
   ruler, not assumed.
2. **Capture the dataset**: 100 photographs of the same object, 10 sessions of
   10, following the frozen split in `dataset/splits.csv`.
3. **Ingest** the photographs: validate, EXIF-correct, resize to 2048×1536,
   quality-check, archive the originals and write a capture log.
4. **Annotate** the object: binary mask and/or polygon, validated and reviewed.
5. **Train** a U-Net + MobileNetV3-Small segmentation model on CPU, 512² input,
   then refine at full resolution.
6. **Measure**: detect the printed ArUco reference, compute the homography to
   the metric plane, take the segmentation mask's oriented min-area rectangle,
   and convert pixels to millimetres. The inference command refuses to emit a
   valid measurement when the calibration, measured reference size, or
   checkpoint is missing.
7. **Prove it**: compare 12 measured instances against a real instrument and
   report the actual error.
8. **Report it**: use the Phase 4A tools for raw-vs-undistorted comparison, the
   measured tilt study, a CPU benchmark, accuracy/per-instance reporting and
   refusal analysis. Each tool writes an explicit `unavailable` record when its
   physical inputs are missing.

---

## Repository layout

```
calibration/        ChArUco board generator + camera calibration pipeline
dataset/            frozen split, ingestion, quality, annotation, reporting
  config/quality.json   committed quality thresholds
  splits.csv            frozen 100-image manifest (never edited)
  report/               generated dataset report (committed baseline: 0/100)
measureit/          shared camera model and image I/O
segmentation/       U-Net + MobileNetV3-Small, training, evaluation, metrics
measurement/        ArUco reference, homography geometry, measurement engine,
                    inference CLI, reference.json template, physical manifests
evaluation/         Phase 4A experiments: raw vs undistorted, condition study,
                    CPU benchmark, accuracy study, refusal analysis
runs/               local training runs and checkpoints (gitignored)
results/            local inference and experiment output (gitignored)
docs/               PLAN.md, OBJECTIVES.md, DEFINITION_OF_DONE.md,
                    DECISION_RECORD.md
tests/              synthetic-only unit tests (never write into dataset/)
tools/              helper scripts
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Windows is the development platform; the tools use `pathlib` and work on any OS.

### Phase 3 environment (verified, not assumed)

The existing Python 3.15 environment is compatible, so no second environment was
created. Verified on this development machine:

| Item | Value |
|---|---|
| Python | 3.15.0 |
| torch / torchvision | 2.15.0+cpu / 0.29.2+cpu (PyPI Windows wheels) |
| CUDA | not available, not built |
| OpenCV | 4.14.0 (contrib, ArUco included) |
| RAM / logical CPUs | 16 GB / 8 (torch uses 4 threads by default) |
| Model parameters | 1,739,505 |
| Checkpoint size | ~6.8 MB (state dict ~6.7 MB) |

The ImageNet encoder weights (~10 MB) are downloaded once, only when training
with `--pretrained` (the default). Use `--no-pretrained` offline.

Measured CPU timings with **random, untrained weights** — architecture timing,
not accuracy:

| Operation | Time |
|---|---|
| U-Net forward, 1×3×512×512 | 0.13 s median (0.11 s best of 5) |
| Coarse + full-resolution refinement, 2048×1536 frame | 0.41 s median |
| ArUco detection + homography + rectangle | ~0.07 s |
| Full inference command | ~1.0 s (includes model and image loading) |

These numbers only say the software fits the CPU budget. They are not a model
quality claim.

---

## The frozen dataset design

`dataset/splits.csv` is a pre-registered manifest of 100 stems:

| Split | Sessions | Images |
|---|---|---|
| train | S01–S07 | 70 |
| val | S08–S09 | 20 |
| test | S10 | 10 |

Sessions never cross splits — this is enforced by the loader, not by convention.
The file is **never written by any tool**. Its SHA-256 is checked before every
dataset command and by a test:

```
1D726C514B6D946F78607B9864C24567E787C88AF7DBF515BBCE3A57FE3DD75C
```

The `.gitattributes` entry forces CRLF on checkout so that the pre-registered
SHA is reproducible on any platform; the repository still stores the canonical
LF blob. If you edit the manifest, every dataset command stops with an error.

### A note on the `condition` column

The `condition` strings (for example `center_500mm_rot30_...`) were frozen in
Phase 0, before the camera and the framing rule were fixed. Some of those
distances are **nominal labels, not capture instructions**. The operative rule
from `docs/PLAN.md` is:

> the object spans **25–30 % of the frame width** (≈280–340 mm for a 120 mm
> plate on the Pixel 8 Pro main lens).

Capture to the framing rule. Record the actual distance and any deviation in the
`note` column of `dataset/capture_log.csv`. The frozen label stays as it is.

### Canonical filenames

Every processed image is `dataset/images/<stem>.jpg` or `<stem>.png`, where
`<stem>` is one of the 100 pre-registered stems (`S01_001` … `S10_010`). The
extension follows the source (JPEG stays JPEG, PNG stays PNG; `.jpeg` is
normalised to `.jpg`).

---

## Phase 2 — ingesting photographs

### 1. Stage the originals

Copy the camera files into `data_incoming/` (gitignored). Do not rename the
processed images by hand; the tool writes those.

### 2. Name each file with its stem, or supply a mapping

Either rename the photos:

```
data_incoming/S01_001.jpg
data_incoming/S01_002.jpg
...
```

or keep the phone names and add `data_incoming/mapping.csv`:

```csv
source,stem
IMG_20261105_101500.jpg,S01_001
IMG_20261105_101530.jpg,S01_002
```

If `mapping.csv` exists it is used automatically; `--map <file>` points at a
mapping elsewhere. Every stem must be in `dataset/splits.csv`; unknown stems are
rejected.

### 3. See which stems are still needed

```powershell
python -m dataset.ingest plan --session S01
python -m dataset.ingest plan --split val
```

### 4. Dry run, then ingest

```powershell
python -m dataset.ingest ingest --dry-run
python -m dataset.ingest ingest
```

What ingest does for each accepted file:

- reads JPEG/PNG, applies EXIF orientation, rejects a wrong aspect ratio instead
  of stretching;
- resizes to **2048×1536** (INTER_AREA), so the working set is uniform;
- measures blur and exposure against `dataset/config/quality.json`;
- rejects byte-identical processed duplicates and flags near-duplicates;
- writes `dataset/images/<stem>.<ext>`;
- copies the untouched original to `dataset/originals/<stem>.<ext>`
  (gitignored) and records its SHA-256;
- appends a row to `dataset/capture_log.csv` with hashes, sizes, quality metrics,
  flags and near-duplicate hits.

Re-running is safe: already-ingested stems are skipped. Use `--replace` to
overwrite, `--limit N` to ingest a prefix, `--strict` to reject any flagged
frame. Hard errors (corrupt file, wrong aspect, unknown stem, duplicate content)
reject that file; the command exits `1` if anything was rejected, `2` for a
usage or split error.

### Quality thresholds (advisory)

Configured in `dataset/config/quality.json`, committed before the first real
capture. They are starting values to be re-tuned against real sessions, and that
change must be recorded in the file's `notes`.

| Check | Initial threshold | Flag |
|---|---|---|
| Blur | variance of Laplacian < 120 | `blurry` |
| Underexposed | mean luma < 45 **or** > 8 % of pixels ≤ 5 | `underexposed` |
| Overexposed | mean luma > 210 **or** > 5 % of pixels ≥ 250 | `overexposed` |
| Exact duplicate | identical processed SHA-256 | rejected |
| Near duplicate | 64-bit dHash Hamming distance ≤ 5 | reported, not rejected |

A flag is a request for review, **not** an automatic deletion. Nothing is
removed from disk by the quality check.

### 5. Regenerate the report

```powershell
python -m dataset.reporting
```

Writes `dataset/report/dataset_report.md`, `inventory.csv` and `quality.csv`
from what is actually on disk: counts by split and session, annotation counts,
missing masks, resolution distribution, quality statistics, duplicate pairs and
annotation problems. `inventory.csv` always lists all 100 stems, present or not.

### Originals and provenance

- `dataset/images/` — committed 2048×1536 working set.
- `dataset/originals/` — untouched camera files, **local only** (large), one per
  stem; their SHA-256 is in `dataset/capture_log.csv`.
- `dataset/capture_log.csv` — append-only ingest history (commit it).
- `data_incoming/` — staging, gitignored.

---

## Phase 2 — annotation

Two interchangeable representations are supported. A human creates one; the
tools derive the other.

### Binary mask

- `dataset/masks/<stem>.png`
- 8-bit grayscale, **2048×1536**, values only 0 (background) and 255 (object).
- Exactly one connected component.
- Object area must be between 0.5 % and 80 % of the frame.

### Polygon JSON

`dataset/annotations/<stem>.json`:

```json
{
  "schema": "measureit.polygon/1",
  "stem": "S01_001",
  "image": "S01_001.jpg",
  "image_size": [2048, 1536],
  "objects": [
    {
      "label": "object",
      "points": [[600, 500], [1100, 500], [1100, 900], [600, 900]]
    }
  ],
  "notes": ""
}
```

Rules enforced by `validate`:

- one `"object"` per image (single-class segmentation);
- at least 3 points, finite, inside the frame, non-self-intersecting,
  area ≥ 100 px²;
- `stem` matches the filename and the image, `image_size` matches the real
  image;
- if both a mask and a polygon exist they must agree at **IoU ≥ 0.98**.

### Commands

```powershell
# check everything on disk
python -m dataset.annotate validate

# human-drawn polygon -> training mask
python -m dataset.annotate polygons-to-masks

# existing mask -> editable polygon (skips existing files unless --force)
python -m dataset.annotate masks-to-polygons

# overlays (polygon red, mask green) + dataset/review/summary.csv
python -m dataset.annotate review
```

Filter any command with `--stem S01_001`. `polygons-to-masks` refuses to
rasterise a polygon without its image or with invalid geometry.
`masks-to-polygons` refuses to annotate an image that is not on disk.

### Manual annotation workflow

There is deliberately no annotation GUI in this repository; use any tool you
trust, then validate:

1. **Mask route** — open `dataset/images/<stem>.jpg` in GIMP/Krita/Photoshop,
   paint the object pure white on a black layer at 2048×1536, export as PNG to
   `dataset/masks/<stem>.png`, then run `masks-to-polygons` to get the editable
   outline and `review` to eyeball it.
2. **Polygon route** — draw the outline in a tool that exports points, write the
   JSON above (or derive a first draft from a mask and edit it), then run
   `polygons-to-masks` and `review`.

Do not hand-write a mask or polygon for an image that does not exist, and do not
copy an annotation from another frame: the object pose differs per frame.

### Review

`dataset/review/<stem>_overlay.png` shows the photo with the mask boundary in
green and the polygon in red, plus the mask area and IoU. `summary.csv` lists
area percentages, IoU and any problem per stem, so a batch can be checked quickly.

---

## Phase 3 — segmentation model

### Architecture and training

- **U-Net** with a **MobileNetV3-Small** encoder (ImageNet weights when
  pretrained), five levels (1/2 … 1/32) and skip connections; decoder from
  scratch. Output is one logit channel at the input resolution.
- Input is RGB, resized to **512×512** (a 4:3 frame is squashed; the mask is
  resized identically and the coarse mask is mapped back before refinement),
  normalised with the ImageNet mean/std.
- Loss is **0.5 × BCEWithLogits + 0.5 × (1 − soft Dice)**, averaged per image.
- Training is deterministic: Python, NumPy and torch seeds are fixed,
  deterministic algorithms are requested, and augmentations (horizontal /
  vertical flip, ±10° rotation, ±10 % brightness/contrast) are a pure function
  of `(seed, epoch, sample index)`.
- Images and masks come **only** from the frozen split. A split with no genuine
  image+mask pairs raises `DatasetUnavailable`; a partially annotated split is
  refused unless `--allow-partial` is passed, and the reduced count is recorded
  in `run.json`. The trainer never reads synthetic fixtures.
- AdamW, `ReduceLROnPlateau`, early stopping on validation IoU. `best.pt` keeps
  the best validation IoU; `last.pt` also has optimizer/scheduler state for
  resumption. A checkpoint stores the architecture, input size, normalisation,
  training config, epoch, validation metrics and split SHA.
- `runs/` is gitignored: checkpoints are local artifacts, not repository
  content.

### Metrics (definitions, not COCO)

- `mean_iou`: mean per-image IoU of the binary mask thresholded at 0.5.
- `precision` / `recall`: pixel-level counts aggregated over the split.
- `map50` / `map50_95`: **single-class mask AP**. Each image contributes one
  prediction ranked by the mean probability inside the predicted mask; a
  prediction is a true positive when its mask IoU reaches the threshold. AP is
  the all-point interpolated area under the precision–recall curve;
  `map50_95` averages AP at IoU 0.50, 0.55, …, 0.95. This is **not** COCO box
  AP and is never presented as such.
- An empty split or a missing checkpoint produces
  `"available": false` with a reason and `null` metrics — never a fabricated
  zero.

### Commands

```powershell
# Train (refuses an empty or synthetic-only split; writes runs/seg/)
python -m segmentation.train --out runs/seg --epochs 40 --batch-size 4

# Evaluate a checkpoint on the frozen test split
python -m segmentation.evaluate --checkpoint runs/seg/best.pt --split test `
    --out runs/seg/eval_test
```

`python -m segmentation.train --help` documents seed, image size, workers,
pretrained weights, partial-split handling and resume.

**No training run has been performed.** There are 0 annotated images, so no
checkpoint, metric or model-quality claim exists.

---

## Phase 3 — measurement engine and inference CLI

### Reference marker

`measurement/reference.json` (schema `measureit.reference/1`) configures the
ArUco dictionary, marker id, the measured physical size, and the validation
limits. The committed file has:

```json
{ "dictionary": "DICT_4X4_50", "marker_id": 0, "marker_size_mm": null }
```

`marker_size_mm` is the **measured side of the printed marker's outer black
square** (the square whose corners OpenCV reports). It is `null` until a marker
is printed and measured with a ruler/caliper. The inference command exits `3`
with `reference_size_unknown` while it is null; no default dimension is assumed.

One marker is detected, validated (single instance, convex, fully inside the
frame, minimum pixel size, maximum side ratio), and its four corners are mapped
to a `marker_size_mm` square. That homography supplies the physical scale and
rectifies the plane, so camera tilt is compensated **if the marker and object
are coplanar**. A single view cannot verify coplanarity; every successful result
carries that statement in `coplanarity_assumption` and `warnings`.

### Geometry

The segmentation contour is transformed into the rectified millimetre frame and
reduced to an oriented minimum-area rectangle. `width_mm ≥ height_mm`; the
reported angle is the long side's direction in the marker frame (0° along the
marker's top edge, increasing towards its right edge, modulo 180°).

### Reliability and refusal

`reliability` is a **rule-based geometry grade** (`high`, `medium`, `low`), not a
learned probability. It rises with a larger marker, lower foreshortening, a
larger object and more room to the frame border. The result includes the
individual `checks` and human-readable `warnings`.

| Outcome | Code | Exit |
|---|---|---|
| measurement produced | — | 0 |
| unreadable image, unwritable output | — | 2 |
| calibration missing/invalid or wrong image size | `missing_calibration`, `invalid_calibration`, `calibration_image_size_mismatch` | 3 |
| reference config missing/invalid or size null | `missing_reference_config`, `invalid_reference_config`, `reference_size_unknown` | 3 |
| checkpoint missing/invalid | `missing_checkpoint`, `invalid_checkpoint` | 3 |
| no reference marker | `missing_reference` | 4 |
| no object mask | `missing_object` | 4 |
| mask touches the border or is split | `partial_occlusion` | 4 |
| degenerate homography/rectangle, implausible size | `invalid_geometry` | 4 |

An unavailable or rejected measurement still writes the structured JSON (with
`"measurement": null` where appropriate), so the reason is machine-readable. It
never emits a millimetre value from a missing calibration, an unmeasured
reference, an untrained checkpoint, or a rejected image.

### Running inference

```powershell
# 1. Print the ArUco marker, measure its outer square, set marker_size_mm.
# 2. Run a real calibration -> calibration/params/camera.json.
# 3. Train or provide a checkpoint -> runs/seg/best.pt.
python -m measurement.infer `
    --image dataset/images/S01_001.jpg `
    --camera calibration/params/camera.json `
    --reference measurement/reference.json `
    --checkpoint runs/seg/best.pt `
    --out results
```

The image must match the calibrated resolution exactly (2048×1536 for this
project); it is undistorted **before** segmentation and marker detection. The
coarse 512² prediction is followed by a full-resolution crop refinement
(`--no-refine` disables it). Outputs are `results/<stem>_result.json` and
`results/<stem>_overlay.png` (marker in blue, mask contour in green, oriented
rectangle and dimensions in red).

### What the synthetic tests prove

`tests/test_measurement_*.py` generate ArUco markers, synthetic 50 mm
references, known perspective transforms and an explicit 100×60 mm object. They
prove detection, homography/rectification, min-area rectangles, mm conversion,
tilt compensation, reliability grading and every refusal path above. They are
**software tests**, not real-world measurement evidence; no physical marker,
camera or instrument was used.

---

## Phase 4A — experiment automation and reporting

Phase 4A is **software infrastructure only**. None of the commands below has
been run against real photographs, a real calibration, a trained checkpoint or
instrument ground truth. Missing inputs produce an explicit `unavailable`
report; the tools never fill a gap with a guessed value.

Reports go under `results/` (gitignored) unless `--out` says otherwise. The
test suite writes only to temporary directories.

### 1. Raw vs undistorted comparison

```powershell
python -m evaluation.raw_vs_undistorted `
    --image dataset/images/S01_001.jpg `
    --camera calibration/params/camera.json `
    --reference measurement/reference.json `
    --checkpoint runs/seg/best.pt `
    --out results/raw_vs_undistorted
```

For each image the tool writes:

- `<stem>_comparison.json` — raw and undistorted measurement records, camera
  metadata, timings and the difference;
- `<stem>_raw.png` and `<stem>_undistorted.png`;
- `<stem>_raw_overlay.png` and `<stem>_undistorted_overlay.png`;
- `<stem>_side_by_side.png`.

The difference is **undistorted minus raw** in millimetres. A positive value
means the undistorted measurement is larger. If either side is rejected, or the
calibration/reference/checkpoint is missing, the difference is `null` and the
report says why. Exit codes match inference: `0` ok, `2` usage, `3` missing
physical data, `4` image-level rejection.

### 2. Tilt / condition study

The study consumes saved inference results and a manifest of **measured** tilt
angles. It does not capture images or create angles.

1. Run inference on every tilt frame into one results directory:

   ```powershell
   python -m measurement.infer --image <frame> --camera <camera.json> `
       --reference <reference.json> --checkpoint <best.pt> --out results/tilt
   ```

2. Write a manifest with the angle measured for each frame:

   ```csv
   stem,measured_angle_deg,target_angle_deg,condition,notes
   T00_01,0.4,0,fixed_400mm_daylight,digital level
   T10_01,10.2,10,fixed_400mm_daylight,digital level
   ```

   `measured_angle_deg` (or `angle_deg`) is required. `target_angle_deg` is
   kept for provenance but is **never** used for grouping. A manifest with only
   targets is refused.

3. Optionally supply manual ground truth for error metrics:

   ```csv
   stem,width_mm,height_mm,instrument,measured_utc,notes
   T00_01,120.1,80.0,digital calliper,2026-11-05,three readings
   ```

   The instrument column is mandatory. The tool never derives ground truth.

4. Generate the report:

   ```powershell
   python -m evaluation.condition_study `
       --manifest measurement/conditions/tilt_measurements.csv `
       --images dataset/images `
       --results results/tilt `
       --ground-truth measurement/ground_truth.csv `
       --out results/condition_study
   ```

Outputs are `condition_study.json`, `condition_study.csv` (per instance) and
`condition_study.md` (grouped by measured angle). Each group reports instance
count, failure/refusal rate, failure codes, MAE width, MAE height, maximum
absolute error and mean percentage error. Missing results count as failures;
error metrics are marked unavailable when no ground-truth pair exists.

### 3. CPU performance benchmark

```powershell
python -m evaluation.timing `
    --image dataset/images/S01_001.jpg `
    --camera calibration/params/camera.json `
    --reference measurement/reference.json `
    --checkpoint runs/seg/best.pt `
    --warmup 2 --repetitions 10 --threads 4 `
    --out results/benchmark
```

Stages timed separately: **model load**, **undistort**, **segmentation**,
**reference marker detection**, **geometry + measurement**, and **total
inference** (undistort through measurement, excluding image load and model
load). The report records the CPU/OS/Python/torch/OpenCV environment, image
dimensions, model input size, warm-up count, repetitions, median, p95, min and
max. `benchmark_repetitions.csv` holds every repetition.

This is a **software timing** report, not an accuracy claim. If the supplied
frame does not produce a valid measurement, the report is marked `partial` and
the geometry stage is left unavailable rather than timing a meaningless value.
Missing calibration, measured reference size or checkpoint produces an
`unavailable` report and exit code `3`.

### 4. Accuracy evaluation reporting

```powershell
python -m evaluation.accuracy_study `
    --ground-truth measurement/ground_truth.csv `
    --results results/accuracy_instances `
    --out results/accuracy
```

The ground-truth CSV is **manual instrument data**, one row per instance:

```csv
stem,width_mm,height_mm,instrument,measured_utc,notes
A01,120.2,80.1,digital calliper,2026-11-05,three readings
```

The accuracy report writes `accuracy.json`, `accuracy_per_instance.csv` and
`accuracy.md`. It reports width, height and combined (width + height pooled)
metrics:

- MAE (mm);
- RMSE (mm);
- mean percentage error (`100 × |error| / ground truth`);
- maximum absolute error (mm);
- failure / refusal rate and a breakdown of failure codes;
- a per-instance comparison table.

Metrics are computed on successful measurements only; the failure rate is
reported separately so a high refusal rate cannot hide behind a good MAE. A
missing ground-truth file, missing results, or zero valid measurements produces
an `unavailable` report and exit code `3`. The tool never generates, estimates
or back-fills a ground-truth value.

### 5. Refusal analysis

```powershell
python -m evaluation.refusal_analysis `
    --cases measurement/conditions/refusal_cases.csv `
    --images dataset/images `
    --results results/refusals `
    --out results/refusal_analysis
```

The cases CSV names the expected outcome for each saved result:

```csv
case_id,stem,expected_code,expected_category,synthetic,notes
R01,ref_missing_01,missing_reference,missing_reference,true,generated marker removed
R02,ref_clipped_02,invalid_geometry,invalid_marker_geometry,true,marker clipped
R03,occluded_03,partial_occlusion,occlusion,true,mask touches border
R04,empty_04,missing_object,unreliable_segmentation,true,no mask
R05,no_camera_05,missing_calibration,invalid_calibration,true,calibration absent
```

A case passes only when both the observed code and the expected category match.
The report covers these required categories: `missing_reference`,
`invalid_marker_geometry`, `occlusion`, `unreliable_segmentation`,
`invalid_calibration` and `excessive_tilt`. It also recognises
`invalid_object_geometry`, `reference_unavailable`, `checkpoint_unavailable`,
`missing_result` and `ok`.

Outputs are `refusal_analysis.json`, `refusal_cases.csv` and
`refusal_analysis.md`. The report shows a coverage table, including required
categories with **zero** cases, and counts synthetic / real / unlabelled cases.
The current engine has **no excessive-tilt threshold**, so that category is
reported as not covered unless a saved result actually carries an
`excessive_tilt` code. Exit codes: `0` all cases match, `1` at least one
mismatch, `2` usage error, `3` no results available.

### 6. Report locations

| Experiment | Default output | Main files |
|---|---|---|
| raw vs undistorted | `results/raw_vs_undistorted/` | `*_comparison.json`, overlays, side-by-side |
| tilt / condition study | `results/condition_study/` | `condition_study.{json,csv,md}` |
| CPU benchmark | `results/benchmark/` | `benchmark.{json,md}`, `benchmark_repetitions.csv` |
| accuracy study | `results/accuracy/` | `accuracy.{json,md}`, `accuracy_per_instance.csv` |
| refusal analysis | `results/refusal_analysis/` | `refusal_analysis.{json,md}`, `refusal_cases.csv` |

`results/` is gitignored: these are local experiment artifacts. The same tools
can write elsewhere with `--out`.

### 7. Synthetic tests versus physical validation

- Every Phase 4A test fixture is generated in a temporary directory. Markers,
  cameras, masks, angles and ground truth in the tests are **synthetic**.
- A refusal-analysis case is evidence only for what its `synthetic` column
  says. If no case is labelled real, the report states that it is not
  real-world evidence.
- No tool invents a ground-truth dimension, a measured tilt angle, a benchmark
  timing or an accuracy metric. Missing physical data is reported as
  unavailable.
- Passing the 4A tests proves that the reporting and refusal plumbing works; it
  does not prove measurement accuracy, robustness or latency on real data.

### 8. Phase 4A limitations

- No real calibration, image, mask, checkpoint, instrument measurement or tilt
  angle exists yet, so no Phase 4A report can contain a real result.
- The tilt study requires a measured angle per frame; a target angle is not a
  measurement.
- The benchmark is machine- and thread-count-specific and says nothing about
  accuracy.
- Accuracy metrics describe successful measurements only; read the failure /
  refusal rate with them.
- Excessive tilt is not implemented as a refusal threshold. That remains an
  open decision to be justified by real tilt data.
- Synthetic fixtures are software checks, not physical evidence.

---

## Adding your Pixel 8 Pro photographs later

The camera is fixed in `docs/PLAN.md`: **Pixel 8 Pro, main (wide) lens, 1.0×
zoom, 4:3, 2048×1536 working resolution**, focus/ISO/white balance locked per
session.

1. Print the ChArUco board (`python -m calibration.make_board generate`) on A4 at
   100 % scale, measure the printed square with a ruler, and record it:
   `python -m calibration.make_board measure --square-x-mm <x> --square-y-mm <y>`.
   The calibration reads the measured square from `board_spec.json`; it is never
   assumed to be exactly 25 mm.
2. Generate and print the reference marker (default `DICT_4X4_50`, id 0), then
   measure its **outer black square** with a ruler/caliper and put that value in
   `measurement/reference.json` as `marker_size_mm`:

   ```powershell
   python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); cv2.imwrite('reference_marker.png', cv2.aruco.generateImageMarker(d, 0, 1200))"
   ```

   Do not use a nominal paper size as the marker size: measure the print.
3. In the Pixel camera's Pro controls: main lens, 1.0×, 4:3, 12 MP (not 50 MP),
   flash off, timer 3 s, focus locked on the object plane, ISO and white balance
   fixed for the whole session. Keep the phone in the **same orientation**
   (landscape) for every frame — ingest rejects a rotated frame rather than
   stretching it.
4. Frame the object at **25–30 % of the frame width** (≈280–340 mm for a 120 mm
   plate). Keep the marker and the object on the **same plane**; the measurement
   assumes coplanarity and cannot verify it from one view.
5. Capture 30+ board views, then run the real calibration:
   `python -m calibration.calibrate`. Only a passing run may write
   `calibration/params/camera.json` and `calibration/report/`.
6. Capture each session (10 frames), transfer the originals to `data_incoming/`,
   name them with their stems (or add `mapping.csv`), then:
   `python -m dataset.ingest ingest`, `python -m dataset.annotate validate`,
   `python -m dataset.annotate review`, `python -m dataset.reporting`.
7. Train and evaluate on the annotated split, then measure a new frame:

   ```powershell
   python -m segmentation.train --out runs/seg --epochs 40
   python -m segmentation.evaluate --checkpoint runs/seg/best.pt --split test --out runs/seg/eval_test
   python -m measurement.infer --image dataset/images/S01_001.jpg `
       --camera calibration/params/camera.json `
       --reference measurement/reference.json `
       --checkpoint runs/seg/best.pt --out results
   ```

8. Commit `dataset/images/`, `dataset/masks/` (or `dataset/annotations/`),
   `dataset/capture_log.csv` and the regenerated `dataset/report/`. Do not commit
   `dataset/originals/`, `data_incoming/`, `runs/` or `results/`.

Until that happens the dataset is **not** complete and no measurement claim is
made.

---

## Command reference

| Command | Purpose |
|---|---|
| `python -m calibration.make_board` | generate/measure the ChArUco board |
| `python -m calibration.calibrate` | real calibration run with gates |
| `python -m dataset.ingest plan` | list stems still missing an image |
| `python -m dataset.ingest ingest` | validate, normalise, archive, log |
| `python -m dataset.annotate validate` | check masks/polygons on disk |
| `python -m dataset.annotate polygons-to-masks` | rasterise polygon JSON |
| `python -m dataset.annotate masks-to-polygons` | recover polygons from masks |
| `python -m dataset.annotate review` | overlays + summary CSV |
| `python -m dataset.reporting` | regenerate the dataset report |
| `python -m segmentation.train` | train the U-Net on the frozen splits |
| `python -m segmentation.evaluate` | evaluate a checkpoint, or report unavailable |
| `python -m measurement.infer` | measure one image, or refuse explicitly |
| `python -m evaluation.raw_vs_undistorted` | measure raw and undistorted versions of one frame |
| `python -m evaluation.condition_study` | group saved results by measured tilt angle |
| `python -m evaluation.timing` | benchmark CPU inference stages |
| `python -m evaluation.accuracy_study` | compare saved measurements with manual ground truth |
| `python -m evaluation.refusal_analysis` | check refusal codes and category coverage |

All commands accept `--help`.

---

## Tests

```powershell
python -m pytest -v
```

The suite is synthetic-only: every fixture is written to a temporary directory,
never into `dataset/`. It covers EXIF orientation, aspect rejection, resolution
normalisation, duplicate content, split integrity and the frozen SHA, quality
thresholds, mask validation, polygon/mask round-trips, the IoU gate, corrupted
files, and the CLIs end to end. Phase 3 adds model/loss/checkpoint tests, metric
tests (including the mask-AP definition), frozen-split loader tests, generated
ArUco detection, homography/rectification, min-area rectangles, pixel-to-mm
conversion, tilt compensation and rejection tests for missing reference,
missing object, partial occlusion, invalid geometry and every CLI refusal path.

Phase 4A adds tests for strict CSV schemas and manual-ground-truth provenance,
raw-vs-undistorted difference arithmetic and unavailable reports, measured-angle
grouping (including refusal of target-only manifests), condition-study metrics,
staged benchmark timing and thread restoration, accuracy metrics from known
synthetic numbers, refusal-code/category matching and coverage gaps. As of this
commit there are **187 tests** and all of them pass (`187 passed`).

---

## What is not claimed

- No camera has been calibrated; there are no intrinsics in the repository.
- No real dataset image, mask or polygon exists; the 100-image dataset is not
  complete.
- No reference marker has been printed or measured; `marker_size_mm` is `null`.
- No model has been trained; there is no checkpoint and no measured model metric.
- No measurement has been compared with an instrument, so no accuracy figure is
  claimed.
- No Phase 4A report has been produced from real artifacts. The tools are
  implemented and tested; they are not results.
- No tilt/condition study, no real-artifact benchmark and no real refusal study
  exists. Reports generated from synthetic fixtures are labelled as such and are
  not accuracy or robustness evidence.
- Tilt angles, marker sizes and object sizes in the tests are synthetic. The
  tests do not demonstrate real-world measurement accuracy.
- The measurement assumes the marker and object are coplanar. That assumption is
  stated in every result and is not verified by the software.
- `reliability` is a rule-based geometry grade, not a calibrated probability or a
  confidence interval.
- Excessive tilt is not implemented as a refusal threshold; the refusal report
  records that gap rather than inventing one.

See `docs/PLAN.md`, `docs/OBJECTIVES.md` and `docs/DEFINITION_OF_DONE.md` for the
contract this work is measured against.
