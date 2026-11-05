"""Read-only view of what is actually on disk in ``dataset/``.

Used by the annotation tools and by the report so both describe the same
state. Missing images are reported as missing; nothing is created here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2

from dataset.annotations import (
    AnnotationError,
    load_mask,
    load_polygon,
    mask_iou,
    polygon_area,
    polygon_to_mask,
    validate_mask,
    validate_polygon,
)
from dataset.quality import (
    QualityConfig,
    QualityMetrics,
    assess,
    dhash64,
)
from dataset.splits import SplitRow, image_for_stem

MASK_POLYGON_IOU_MIN = 0.98


@dataclass
class StemState:
    stem: str
    split: str
    session: str
    condition: str
    image: Path | None = None
    image_size: tuple[int, int] | None = None
    mask_path: Path | None = None
    polygon_path: Path | None = None
    mask_area_pct: float | None = None
    polygon_area_pct: float | None = None
    iou: float | None = None
    quality: QualityMetrics | None = None
    dhash: int | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def has_annotation(self) -> bool:
        return self.mask_path is not None or self.polygon_path is not None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None


@dataclass
class DatasetState:
    stems: list[StemState]
    unknown: list[str]

    @property
    def problems(self) -> list[str]:
        messages = list(self.unknown)
        for state in self.stems:
            messages.extend(f"{state.stem}: {problem}" for problem in state.problems)
        return messages

    @property
    def present(self) -> list[StemState]:
        return [state for state in self.stems if state.image is not None]


def _unknown_stems(directory: Path, stems: set[str], label: str) -> list[str]:
    if not directory.is_dir():
        return []
    found: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        if path.stem not in stems:
            found.append(f"{label} {path.name} has no stem in the frozen split")
    return found


def collect_state(
    dataset_dir: str | Path,
    rows: Iterable[SplitRow],
    config: QualityConfig,
) -> DatasetState:
    root = Path(dataset_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"
    annotations_dir = root / "annotations"

    rows = list(rows)
    catalog = {row.stem: row for row in rows}
    unknown = (
        _unknown_stems(images_dir, set(catalog), "image")
        + _unknown_stems(masks_dir, set(catalog), "mask")
        + _unknown_stems(annotations_dir, set(catalog), "annotation")
    )

    states: list[StemState] = []
    for row in rows:
        state = StemState(
            stem=row.stem, split=row.split, session=row.session, condition=row.condition
        )

        image_path = image_for_stem(images_dir, row.stem)
        gray = None
        if image_path is not None:
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                state.problems.append(f"cannot read image {image_path.name}")
            else:
                state.image = image_path
                state.image_size = (gray.shape[1], gray.shape[0])
                state.quality = assess(gray, config)
                state.dhash = dhash64(gray)

        mask_path = masks_dir / f"{row.stem}.png"
        mask = None
        if mask_path.is_file():
            try:
                mask = load_mask(mask_path)
                state.mask_path = mask_path
            except AnnotationError as exc:
                state.problems.append(f"{mask_path.name}: {exc}")

        annotation_path = annotations_dir / f"{row.stem}.json"
        polygons = None
        if annotation_path.is_file():
            try:
                annotation = load_polygon(annotation_path)
                expected_size = state.image_size or annotation.image_size
                state.problems.extend(
                    validate_polygon(
                        annotation, expected_size=expected_size, expected_stem=row.stem
                    )
                )
                polygons = annotation
                state.polygon_path = annotation_path
                if annotation.objects:
                    state.polygon_area_pct = (
                        polygon_area(annotation.objects[0].points)
                        / (annotation.image_size[0] * annotation.image_size[1])
                        * 100.0
                    )
            except AnnotationError as exc:
                state.problems.append(f"{annotation_path.name}: {exc}")

        if state.image is None and (mask_path.is_file() or annotation_path.is_file()):
            state.problems.append("annotation or mask exists without an image")

        if mask is not None:
            expected_size = state.image_size or config.working_size
            state.problems.extend(
                validate_mask(mask, expected_size=expected_size, config=config)
            )
            state.mask_area_pct = float((mask > 0).mean() * 100.0)

        if mask is not None and polygons is not None and polygons.objects:
            if state.image_size is not None:
                try:
                    polygon_mask = polygon_to_mask(
                        polygons.objects[0].points, state.image_size
                    )
                    state.iou = mask_iou(mask, polygon_mask)
                    if state.iou < MASK_POLYGON_IOU_MIN:
                        state.problems.append(
                            f"mask and polygon disagree (IoU {state.iou:.3f} < "
                            f"{MASK_POLYGON_IOU_MIN:.2f})"
                        )
                except AnnotationError as exc:
                    state.problems.append(str(exc))

        states.append(state)

    return DatasetState(stems=states, unknown=unknown)
