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
| Dataset images | **0 / 100** — no real photographs have been captured yet |
| Annotations | **0** — no masks or polygons exist |
| Segmentation model | not started, not trained |
| Measurement accuracy vs instrument | not measured |

`calibration/params/camera.json` and `calibration/report/` **do not exist**. They
are only written by a real passing calibration run, and no such run has
happened. Nothing in this repository fabricates images, masks, measurements or
metrics. `dataset/report/dataset_report.md` currently says `0 / 100`, which is
the honest state.

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
   and convert pixels to millimetres.
7. **Prove it**: compare 12 measured instances against a real instrument and
   report the actual error.

---

## Repository layout

```
calibration/        ChArUco board generator + camera calibration pipeline
dataset/            frozen split, ingestion, quality, annotation, reporting
  config/quality.json   committed quality thresholds
  splits.csv            frozen 100-image manifest (never edited)
  report/               generated dataset report (committed baseline: 0/100)
measureit/          shared camera model and image I/O
measurement/        accuracy-instance and tilt manifests (physical work pending)
docs/               PLAN.md, OBJECTIVES.md, DEFINITION_OF_DONE.md
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

## Adding your Pixel 8 Pro photographs later

The camera is fixed in `docs/PLAN.md`: **Pixel 8 Pro, main (wide) lens, 1.0×
zoom, 4:3, 2048×1536 working resolution**, focus/ISO/white balance locked per
session.

1. Print the ChArUco board (`python -m calibration.make_board generate`) on A4 at
   100 % scale, measure the printed square with a ruler, and record it:
   `python -m calibration.make_board measure --square-x-mm <x> --square-y-mm <y>`.
   The calibration reads the measured square from `board_spec.json`; it is never
   assumed to be exactly 25 mm.
2. In the Pixel camera's Pro controls: main lens, 1.0×, 4:3, 12 MP (not 50 MP),
   flash off, timer 3 s, focus locked on the object plane, ISO and white balance
   fixed for the whole session. Keep the phone in the **same orientation**
   (landscape) for every frame — ingest rejects a rotated frame rather than
   stretching it.
3. Frame the object at **25–30 % of the frame width** (≈280–340 mm for a 120 mm
   plate). Keep the board and object on the same plane.
4. Capture 30+ board views, then run the real calibration:
   `python -m calibration.calibrate`. Only a passing run may write
   `calibration/params/camera.json` and `calibration/report/`.
5. Capture each session (10 frames), transfer the originals to `data_incoming/`,
   name them with their stems (or add `mapping.csv`), then:
   `python -m dataset.ingest ingest`, `python -m dataset.annotate validate`,
   `python -m dataset.annotate review`, `python -m dataset.reporting`.
6. Commit `dataset/images/`, `dataset/masks/` (or `dataset/annotations/`),
   `dataset/capture_log.csv` and the regenerated `dataset/report/`. Do not commit
   `dataset/originals/` or `data_incoming/`.

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
files, and the CLIs end to end. As of this commit there are 64 tests and all of
them pass (`64 passed`).

---

## What is not claimed

- No camera has been calibrated; there are no intrinsics in the repository.
- No real dataset image, mask or polygon exists; the 100-image dataset is not
  complete.
- No model has been trained.
- No measurement has been compared with an instrument, so no accuracy figure is
  claimed.

See `docs/PLAN.md`, `docs/OBJECTIVES.md` and `docs/DEFINITION_OF_DONE.md` for the
contract this work is measured against.
