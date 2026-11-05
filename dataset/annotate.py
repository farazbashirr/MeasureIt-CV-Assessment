"""Validate annotations, convert polygons to masks and back, build review overlays.

The tools never create an annotation from nothing. ``polygons-to-masks``
rasterises a polygon a human drew; ``masks-to-polygons`` recovers an editable
outline from an existing mask; ``review`` draws what is on disk next to the
photo. An image with no mask and no polygon is reported as unannotated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import cv2

from dataset.annotations import (
    AnnotationError,
    ObjectPolygon,
    PolygonAnnotation,
    draw_review_overlay,
    load_mask,
    load_polygon,
    mask_to_polygon,
    polygon_to_mask,
    validate_polygon,
    write_polygon,
)
from dataset.quality import QualityConfig
from dataset.splits import DEFAULT_SPLITS, load_splits
from dataset.state import DatasetState, StemState, collect_state


def _load_state(args: argparse.Namespace) -> DatasetState:
    rows = load_splits(args.splits)
    config = QualityConfig.load(args.config)
    return collect_state(args.dataset, rows, config)


def _selected(state: DatasetState, stem: str | None) -> list[StemState]:
    if stem is None:
        return state.stems
    chosen = [item for item in state.stems if item.stem == stem]
    if not chosen:
        raise SystemExit(f"ERROR: stem {stem!r} is not in the frozen split")
    return chosen


def _print_counts(state: DatasetState) -> None:
    present = state.present
    masks = [item for item in present if item.has_mask]
    polygons = [item for item in present if item.polygon_path is not None]
    annotated = [item for item in present if item.has_annotation]
    print(f"images present : {len(present)}")
    print(f"masks          : {len(masks)}")
    print(f"polygons       : {len(polygons)}")
    print(f"unannotated    : {len(present) - len(annotated)}")


def _validate(args: argparse.Namespace) -> int:
    state = _load_state(args)
    _print_counts(state)
    problems = state.problems
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"problems       : {len(problems)}")
    return 1 if problems else 0


def _polygons_to_masks(args: argparse.Namespace) -> int:
    state = _load_state(args)
    written = skipped = 0
    problems: list[str] = []

    for item in _selected(state, args.stem):
        if item.polygon_path is None:
            continue
        if item.image is None:
            problems.append(f"{item.stem}: image missing, refusing to rasterise")
            continue
        try:
            annotation = load_polygon(item.polygon_path)
            issues = validate_polygon(
                annotation, expected_size=item.image_size, expected_stem=item.stem
            )
            if issues:
                problems.append(f"{item.stem}: {issues[0]}")
                continue
            target = args.dataset / "masks" / f"{item.stem}.png"
            if target.exists() and not args.force:
                skipped += 1
                continue
            mask = polygon_to_mask(annotation.objects[0].points, item.image_size)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), mask):
                problems.append(f"{item.stem}: could not write {target}")
                continue
            written += 1
        except AnnotationError as exc:
            problems.append(f"{item.stem}: {exc}")

    print(f"written {written}, skipped {skipped}, problems {len(problems)}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


def _masks_to_polygons(args: argparse.Namespace) -> int:
    state = _load_state(args)
    written = skipped = 0
    problems: list[str] = []

    for item in _selected(state, args.stem):
        if item.mask_path is None:
            continue
        if item.image is None or item.image_size is None:
            problems.append(f"{item.stem}: image missing, refusing to annotate")
            continue
        target = args.dataset / "annotations" / f"{item.stem}.json"
        if target.exists() and not args.force:
            skipped += 1
            continue
        try:
            mask = load_mask(item.mask_path)
            points = mask_to_polygon(mask, epsilon_px=args.epsilon)
            annotation = PolygonAnnotation(
                stem=item.stem,
                image=item.image.name,
                image_size=item.image_size,
                objects=(ObjectPolygon(label="object", points=tuple(points)),),
                notes="derived from mask by dataset.annotate masks-to-polygons",
            )
            write_polygon(annotation, target)
            written += 1
        except AnnotationError as exc:
            problems.append(f"{item.stem}: {exc}")

    print(f"written {written}, skipped {skipped}, problems {len(problems)}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


def _review(args: argparse.Namespace) -> int:
    state = _load_state(args)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    written = 0

    for item in _selected(state, args.stem):
        if not item.has_annotation:
            continue
        if item.image is None:
            problems.append(f"{item.stem}: annotation without an image")
            continue

        image = cv2.imread(str(item.image), cv2.IMREAD_COLOR)
        if image is None:
            problems.append(f"{item.stem}: cannot read {item.image.name}")
            continue

        mask = load_mask(item.mask_path) if item.mask_path is not None else None
        points = None
        if item.polygon_path is not None:
            try:
                annotation = load_polygon(item.polygon_path)
                points = annotation.objects[0].as_array() if annotation.objects else None
            except AnnotationError as exc:
                problems.append(f"{item.stem}: {exc}")

        overlay = draw_review_overlay(
            image, mask=mask, polygon=points, stem=item.stem, iou=item.iou
        )
        target = out_dir / f"{item.stem}_overlay.png"
        if not cv2.imwrite(str(target), overlay):
            problems.append(f"{item.stem}: could not write {target}")
            continue
        written += 1
        rows.append(
            {
                "stem": item.stem,
                "image": item.image.name,
                "width": str(item.image_size[0]),
                "height": str(item.image_size[1]),
                "mask": str(item.mask_path is not None).lower(),
                "polygon": str(item.polygon_path is not None).lower(),
                "mask_area_pct": "" if item.mask_area_pct is None else f"{item.mask_area_pct:.3f}",
                "polygon_area_pct": (
                    "" if item.polygon_area_pct is None else f"{item.polygon_area_pct:.3f}"
                ),
                "iou": "" if item.iou is None else f"{item.iou:.4f}",
                "problems": "; ".join(item.problems),
            }
        )

    summary = out_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stem",
                "image",
                "width",
                "height",
                "mask",
                "polygon",
                "mask_area_pct",
                "polygon_area_pct",
                "iou",
                "problems",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"overlays written {written} to {out_dir}")
    print(f"summary: {summary}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.annotate",
        description="Validate and convert masks and polygons for the frozen split.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dataset", default=Path("dataset"), type=Path)
        command.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
        command.add_argument("--config", default=Path("dataset/config/quality.json"), type=Path)
        command.add_argument("--stem", default=None, help="only this stem")

    common(sub.add_parser("validate", help="check every mask and polygon on disk"))

    to_masks = sub.add_parser("polygons-to-masks", help="rasterise polygon JSON")
    common(to_masks)
    to_masks.add_argument("--force", action="store_true", help="overwrite an existing mask")

    to_polygons = sub.add_parser("masks-to-polygons", help="recover outlines from masks")
    common(to_polygons)
    to_polygons.add_argument("--force", action="store_true", help="overwrite existing JSON")
    to_polygons.add_argument(
        "--epsilon", type=float, default=1.0, help="approxPolyDP tolerance in pixels"
    )

    review = sub.add_parser("review", help="write overlay images and a summary CSV")
    common(review)
    review.add_argument("--out", default=Path("dataset/review"), type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "polygons-to-masks":
            return _polygons_to_masks(args)
        if args.command == "masks-to-polygons":
            return _masks_to_polygons(args)
        return _review(args)
    except (AnnotationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
