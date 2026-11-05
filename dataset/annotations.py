"""Annotation format: binary masks and polygon JSON.

Two representations are supported and are interchangeable:

* mask  ``dataset/masks/<stem>.png``, 8-bit, 2048x1536, only 0 and 255;
* polygon  ``dataset/annotations/<stem>.json``, schema ``measureit.polygon/1``.

The polygon file is the editable source when a human draws the outline; the
mask is what training consumes. ``masks-to-polygons`` can recover an editable
polygon from a mask, which is how a mask produced by another tool enters the
review cycle. Neither direction invents an object: an empty or missing mask is
reported as missing, never filled in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from dataset.quality import QualityConfig

POLYGON_SCHEMA = "measureit.polygon/1"
LABEL = "object"
MIN_POLYGON_AREA_PX = 100.0


class AnnotationError(RuntimeError):
    """An annotation is missing, malformed or inconsistent with its image."""


@dataclass(frozen=True)
class ObjectPolygon:
    label: str
    points: tuple[tuple[float, float], ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64).reshape(-1, 2)


@dataclass(frozen=True)
class PolygonAnnotation:
    stem: str
    image: str
    image_size: tuple[int, int]
    objects: tuple[ObjectPolygon, ...]
    notes: str = ""


def load_polygon(path: str | Path) -> PolygonAnnotation:
    source = Path(path)
    if not source.is_file():
        raise AnnotationError(f"annotation not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{source} is not valid JSON: {exc}") from exc
    return polygon_from_dict(payload, source=source)


def polygon_from_dict(payload: dict[str, Any], *, source: Path | None = None) -> PolygonAnnotation:
    label = str(source) if source else "polygon"
    if payload.get("schema") != POLYGON_SCHEMA:
        raise AnnotationError(f"{label}: unsupported schema {payload.get('schema')!r}")

    try:
        stem = str(payload["stem"])
        image = str(payload["image"])
        raw_size = payload["image_size"]
        raw_objects = payload["objects"]
    except KeyError as exc:
        raise AnnotationError(f"{label}: missing key {exc.args[0]!r}") from exc

    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise AnnotationError(f"{label}: image_size must be [width, height]")
    if not isinstance(raw_objects, list):
        raise AnnotationError(f"{label}: objects must be a list")

    objects: list[ObjectPolygon] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise AnnotationError(f"{label}: object {index} must be an object")
        try:
            object_label = str(item["label"])
            raw_points = item["points"]
        except KeyError as exc:
            raise AnnotationError(f"{label}: object {index} is missing {exc.args[0]!r}") from exc
        if not isinstance(raw_points, list):
            raise AnnotationError(f"{label}: object {index} points must be a list")
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise AnnotationError(f"{label}: object {index} has a non-pair point")
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise AnnotationError(f"{label}: object {index} has a non-numeric point") from exc
            points.append((x, y))
        objects.append(ObjectPolygon(label=object_label, points=tuple(points)))

    return PolygonAnnotation(
        stem=stem,
        image=image,
        image_size=(int(raw_size[0]), int(raw_size[1])),
        objects=tuple(objects),
        notes=str(payload.get("notes", "")),
    )


def annotation_to_dict(annotation: PolygonAnnotation) -> dict[str, Any]:
    return {
        "schema": POLYGON_SCHEMA,
        "stem": annotation.stem,
        "image": annotation.image,
        "image_size": [annotation.image_size[0], annotation.image_size[1]],
        "objects": [
            {"label": item.label, "points": [[x, y] for x, y in item.points]}
            for item in annotation.objects
        ],
        "notes": annotation.notes,
    }


def polygon_area(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) < 3:
        return 0.0
    x = array[:, 0]
    y = array[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return int(np.sign(value))


def _proper_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def is_self_intersecting(points: Sequence[Sequence[float]] | np.ndarray) -> bool:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(array)
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _proper_cross(array[i], array[(i + 1) % count], array[j], array[(j + 1) % count]):
                return True
    return False


def validate_polygon(
    annotation: PolygonAnnotation,
    *,
    expected_size: tuple[int, int] | None = None,
    expected_stem: str | None = None,
) -> list[str]:
    problems: list[str] = []

    if expected_stem is not None and annotation.stem != expected_stem:
        problems.append(f"stem {annotation.stem!r} != {expected_stem!r}")
    if Path(annotation.image).stem != annotation.stem:
        problems.append(f"image {annotation.image!r} does not belong to stem {annotation.stem!r}")
    if expected_size is not None and tuple(annotation.image_size) != tuple(expected_size):
        problems.append(
            f"image_size {annotation.image_size} != actual {tuple(expected_size)}"
        )

    width, height = annotation.image_size
    if width <= 0 or height <= 0:
        problems.append("image_size must be positive")
        return problems

    if len(annotation.objects) != 1:
        problems.append(f"exactly one object is required, found {len(annotation.objects)}")
        return problems

    item = annotation.objects[0]
    if item.label != LABEL:
        problems.append(f"label {item.label!r} != {LABEL!r}")

    if len(item.points) < 3:
        problems.append(f"a polygon needs at least 3 points, found {len(item.points)}")
        return problems

    array = item.as_array()
    if not np.isfinite(array).all():
        problems.append("polygon has non-finite coordinates")
        return problems
    if array[:, 0].min() < 0 or array[:, 0].max() > width - 1:
        problems.append(f"x outside 0..{width - 1}")
    if array[:, 1].min() < 0 or array[:, 1].max() > height - 1:
        problems.append(f"y outside 0..{height - 1}")
    if problems:
        return problems

    area = polygon_area(array)
    if area < MIN_POLYGON_AREA_PX:
        problems.append(f"polygon area {area:.1f} px is below {MIN_POLYGON_AREA_PX:.0f} px")
    if is_self_intersecting(array):
        problems.append("polygon is self-intersecting")
    return problems


def polygon_to_mask(
    points: Sequence[Sequence[float]] | np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) < 3:
        raise AnnotationError("a polygon needs at least 3 points")
    cv2.fillPoly(mask, [np.round(array).astype(np.int32)], 255, lineType=cv2.LINE_8)
    return mask


def load_mask(path: str | Path) -> np.ndarray:
    source = Path(path)
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise AnnotationError(f"cannot read mask: {source}")
    return mask


def validate_mask(
    mask: np.ndarray,
    *,
    expected_size: tuple[int, int],
    config: QualityConfig,
) -> list[str]:
    problems: list[str] = []
    if mask.ndim != 2:
        problems.append(f"mask must be single-channel, got shape {mask.shape}")
        return problems
    height, width = mask.shape
    if (width, height) != tuple(expected_size):
        problems.append(f"mask is {width}x{height}, expected {expected_size[0]}x{expected_size[1]}")
        return problems

    values = np.unique(mask)
    if not set(int(v) for v in values).issubset({0, 255}):
        problems.append(f"mask values are {values[:8].tolist()}..., expected only 0 and 255")
        return problems

    area_pct = float(np.count_nonzero(mask) / mask.size * 100.0)
    if area_pct < config.mask_area_pct_min:
        problems.append(
            f"mask covers {area_pct:.2f}% (< {config.mask_area_pct_min:.2f}%)"
        )
    if area_pct > config.mask_area_pct_max:
        problems.append(
            f"mask covers {area_pct:.2f}% (> {config.mask_area_pct_max:.2f}%)"
        )

    components, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if components - 1 != 1:
        problems.append(f"mask has {components - 1} connected components, expected 1")

    return problems


def mask_to_polygon(mask: np.ndarray, *, epsilon_px: float = 1.0) -> list[tuple[float, float]]:
    if mask.ndim != 2:
        raise AnnotationError("mask must be single-channel")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise AnnotationError("mask has no contour")
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        raise AnnotationError("mask contour has zero area")
    approximated = cv2.approxPolyDP(largest, float(epsilon_px), True).reshape(-1, 2)
    if len(approximated) < 3:
        raise AnnotationError("mask contour has fewer than 3 points after approximation")
    return [(float(x), float(y)) for x, y in approximated]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise AnnotationError(f"mask shapes differ: {left.shape} vs {right.shape}")
    a = left > 0
    b = right > 0
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def draw_review_overlay(
    image: np.ndarray,
    *,
    mask: np.ndarray | None,
    polygon: Sequence[Sequence[float]] | None,
    stem: str,
    iou: float | None = None,
) -> np.ndarray:
    canvas = image.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 220, 0), 2, lineType=cv2.LINE_AA)
    if polygon is not None and len(polygon) >= 3:
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    height, width = canvas.shape[:2]
    caption = stem
    if mask is not None:
        caption += f"  mask {np.count_nonzero(mask) / mask.size * 100.0:.2f}%"
    if iou is not None:
        caption += f"  IoU {iou:.3f}"
    cv2.putText(
        canvas, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        canvas, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    return cv2.resize(canvas, (width // 2, height // 2), interpolation=cv2.INTER_AREA)


def write_polygon(annotation: PolygonAnnotation, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(annotation_to_dict(annotation), indent=2) + "\n", encoding="utf-8"
    )


def iter_annotation_files(directory: str | Path) -> Iterable[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"), key=lambda p: p.name.lower())


def iter_mask_files(directory: str | Path) -> Iterable[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.png"), key=lambda p: p.name.lower())
