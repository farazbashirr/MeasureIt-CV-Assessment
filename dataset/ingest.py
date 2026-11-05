"""Ingest phone photographs into the frozen 100-image split.

The incoming folder is staging only. Each accepted frame is:

1. validated and EXIF-corrected,
2. resized to the working resolution without changing its aspect ratio,
3. quality-checked and duplicate-checked,
4. written as ``dataset/images/<stem>.<ext>``,
5. archived untouched as ``dataset/originals/<stem>.<ext>``,
6. appended to ``dataset/capture_log.csv`` with hashes and measurements.

The split manifest is never written. A stem that is not in
``dataset/splits.csv`` is rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from dataset.quality import (
    QualityConfig,
    assess,
    canonical_suffix,
    content_sha256,
    dhash64,
    format_flags,
    hamming,
    probe_size,
)
from dataset.splits import (
    DEFAULT_SPLITS,
    IMAGE_SUFFIXES,
    SplitError,
    SplitRow,
    by_stem,
    image_for_stem,
    load_splits,
)
from measureit.io import ImageLoadError, load_image

CAPTURE_LOG_FIELDS = [
    "ingested_utc",
    "stem",
    "split",
    "session",
    "condition",
    "source_file",
    "original_file",
    "original_sha256",
    "original_width",
    "original_height",
    "image_file",
    "processed_sha256",
    "width",
    "height",
    "blur_laplacian_var",
    "mean_luma",
    "dark_pct",
    "bright_pct",
    "flags",
    "near_duplicates",
    "note",
]


class IngestError(RuntimeError):
    """The staging folder cannot be ingested as requested."""


@dataclass(frozen=True)
class KnownImage:
    stem: str
    sha256: str
    dhash: int


@dataclass
class IngestOutcome:
    stem: str
    status: str
    message: str
    flags: tuple[str, ...] = ()
    near_duplicates: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise IngestError(f"incoming folder not found: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def _read_mapping(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise IngestError(f"mapping file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "stem"]:
            raise IngestError(
                f"{path} must have exactly the header source,stem; found {reader.fieldnames!r}"
            )
        rows = [(row["source"].strip(), row["stem"].strip()) for row in reader]
    if not rows:
        raise IngestError(f"{path} contains no rows")
    return rows


def resolve_sources(
    incoming: Path,
    known_stems: dict[str, SplitRow],
    mapping_path: Path | None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return (source, stem) pairs and non-fatal warnings."""
    warnings: list[str] = []
    pairs: list[tuple[Path, str]] = []

    if mapping_path is not None:
        seen: dict[str, str] = {}
        for source_name, stem in _read_mapping(mapping_path):
            source = incoming / source_name
            if source.parent != incoming:
                raise IngestError(f"mapping source must be a file in {incoming}: {source_name!r}")
            if not source.is_file():
                raise IngestError(f"mapped source does not exist: {source_name}")
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                raise IngestError(f"mapped source is not a JPEG or PNG: {source_name}")
            previous = seen.get(stem)
            if previous is not None:
                raise IngestError(
                    f"two sources map to {stem}: {previous!r} and {source_name!r}"
                )
            if stem not in known_stems:
                raise IngestError(
                    f"stem {stem!r} is not in the frozen split; check dataset/splits.csv"
                )
            seen[stem] = source_name
            pairs.append((source, stem))

        mapped = {source.name for source, _ in pairs}
        extras = [path.name for path in _image_files(incoming) if path.name not in mapped]
        if extras:
            warnings.append(
                f"{len(extras)} incoming image(s) are not in the mapping and were ignored: "
                + ", ".join(extras[:5])
            )
        return pairs, warnings

    for path in _image_files(incoming):
        stem = path.stem
        if stem not in known_stems:
            raise IngestError(
                f"{path.name}: stem {stem!r} is not in the frozen split. Rename the file "
                "to one of the pre-registered stems (see `python -m dataset.ingest plan`) "
                "or pass --map."
            )
        pairs.append((path, stem))

    stems = [stem for _, stem in pairs]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise IngestError(
            "two incoming files have the same stem: " + ", ".join(duplicates)
        )
    if not pairs:
        raise IngestError(
            f"{incoming} contains no JPEG or PNG files. Copy the original photos there first."
        )
    return pairs, warnings


def scan_known(images_dir: Path) -> dict[str, KnownImage]:
    known: dict[str, KnownImage] = {}
    if not images_dir.is_dir():
        return known
    for path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        known[path.stem] = KnownImage(
            stem=path.stem,
            sha256=content_sha256(path),
            dhash=dhash64(image),
        )
    return known


def _encode(image: np.ndarray, suffix: str) -> bytes:
    if suffix == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise IngestError(f"could not encode {suffix}")
    return buffer.tobytes()


def _near_duplicates(
    digest: int, known: dict[str, KnownImage], stem: str, threshold: int
) -> list[str]:
    hits: list[tuple[int, str]] = []
    for other in known.values():
        if other.stem == stem:
            continue
        distance = hamming(digest, other.dhash)
        if distance <= threshold:
            hits.append((distance, other.stem))
    hits.sort()
    return [f"{other} (hamming {distance})" for distance, other in hits]


def _append_log(log_path: Path, row: dict[str, str]) -> None:
    exists = log_path.is_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def ingest(
    *,
    incoming: Path,
    dataset_dir: Path,
    splits_path: Path,
    config_path: Path,
    mapping_path: Path | None,
    replace: bool,
    strict: bool,
    dry_run: bool,
    limit: int | None = None,
) -> tuple[list[IngestOutcome], list[str]]:
    rows = load_splits(splits_path)
    catalog = by_stem(rows)
    config = QualityConfig.load(config_path)

    images_dir = dataset_dir / "images"
    originals_dir = dataset_dir / "originals"
    masks_dir = dataset_dir / "masks"
    annotations_dir = dataset_dir / "annotations"
    if not dry_run:
        for directory in (images_dir, originals_dir, masks_dir, annotations_dir):
            directory.mkdir(parents=True, exist_ok=True)

    pairs, warnings = resolve_sources(incoming, catalog, mapping_path)
    if limit is not None:
        dropped = len(pairs) - limit if len(pairs) > limit else 0
        if dropped:
            warnings.append(f"--limit {limit}: {dropped} source(s) left for the next run")
        pairs = pairs[:limit]
    known = scan_known(images_dir)

    outcomes: list[IngestOutcome] = []
    for source, stem in pairs:
        row = catalog[stem]
        destination = image_for_stem(images_dir, stem)

        if destination is not None and not replace:
            outcomes.append(
                IngestOutcome(stem, "skipped", f"already ingested as {destination.name}")
            )
            continue

        try:
            image = load_image(source, config.working_size)
        except ImageLoadError as exc:
            outcomes.append(IngestOutcome(stem, "rejected", str(exc)))
            continue

        metrics = assess(image, config)
        suffix = canonical_suffix(source)
        encoded = _encode(image, suffix)
        processed_sha = hashlib.sha256(encoded).hexdigest().upper()
        digest = dhash64(image)

        exact = next(
            (
                other
                for other in known.values()
                if other.sha256 == processed_sha and other.stem != stem
            ),
            None,
        )
        if exact is not None:
            outcomes.append(
                IngestOutcome(
                    stem,
                    "rejected",
                    f"identical processed content to already-ingested {exact.stem}",
                    flags=metrics.flags,
                )
            )
            continue

        near = _near_duplicates(digest, known, stem, config.hamming_max)
        if strict and metrics.flags:
            outcomes.append(
                IngestOutcome(
                    stem,
                    "rejected",
                    f"quality flags in --strict mode: {', '.join(metrics.flags)}",
                    flags=metrics.flags,
                    near_duplicates=tuple(near),
                )
            )
            continue

        if dry_run:
            outcomes.append(
                IngestOutcome(
                    stem,
                    "would-ingest",
                    f"{source.name} -> {stem}{suffix}",
                    flags=metrics.flags,
                    near_duplicates=tuple(near),
                )
            )
            continue

        try:
            original_width, original_height = probe_size(source)
        except Exception as exc:  # pragma: no cover - load_image already decoded it
            outcomes.append(IngestOutcome(stem, "rejected", f"cannot read size: {exc}"))
            continue

        if destination is not None:
            destination.unlink()
        destination = images_dir / f"{stem}{suffix}"
        destination.write_bytes(encoded)

        archived = originals_dir / f"{stem}{source.suffix.lower()}"
        shutil.copy2(source, archived)

        _append_log(
            dataset_dir / "capture_log.csv",
            {
                "ingested_utc": _utc_now(),
                "stem": stem,
                "split": row.split,
                "session": row.session,
                "condition": row.condition,
                "source_file": source.name,
                "original_file": archived.relative_to(dataset_dir).as_posix(),
                "original_sha256": content_sha256(archived),
                "original_width": original_width,
                "original_height": original_height,
                "image_file": destination.relative_to(dataset_dir).as_posix(),
                "processed_sha256": processed_sha,
                "width": image.shape[1],
                "height": image.shape[0],
                "blur_laplacian_var": f"{metrics.blur_variance:.2f}",
                "mean_luma": f"{metrics.mean_luma:.2f}",
                "dark_pct": f"{metrics.dark_pct:.2f}",
                "bright_pct": f"{metrics.bright_pct:.2f}",
                "flags": format_flags(metrics.flags),
                "near_duplicates": ";".join(near),
                "note": "",
            },
        )

        known[stem] = KnownImage(stem=stem, sha256=processed_sha, dhash=digest)
        outcomes.append(
            IngestOutcome(
                stem,
                "ingested",
                f"{source.name} -> {destination.name}",
                flags=metrics.flags,
                near_duplicates=tuple(near),
            )
        )

    return outcomes, warnings


def _print_outcomes(outcomes: Sequence[IngestOutcome]) -> None:
    for outcome in outcomes:
        flags = f"  flags: {', '.join(outcome.flags)}" if outcome.flags else ""
        near = (
            f"  near: {'; '.join(outcome.near_duplicates)}"
            if outcome.near_duplicates
            else ""
        )
        print(f"  {outcome.status:12s} {outcome.stem:10s} {outcome.message}{flags}{near}")


def _summary(outcomes: Sequence[IngestOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.ingest",
        description="Validate, normalise and file photographs into the frozen split.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("ingest", help="ingest images from the staging folder")
    run.add_argument("--incoming", default=Path("data_incoming"), type=Path)
    run.add_argument("--dataset", default=Path("dataset"), type=Path)
    run.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
    run.add_argument("--config", default=Path("dataset/config/quality.json"), type=Path)
    run.add_argument(
        "--map",
        default=None,
        type=Path,
        help="CSV with source,stem columns; defaults to <incoming>/mapping.csv if present",
    )
    run.add_argument("--replace", action="store_true", help="overwrite an image already ingested")
    run.add_argument("--strict", action="store_true", help="reject any image with a quality flag")
    run.add_argument("--dry-run", action="store_true", help="validate without writing")
    run.add_argument("--limit", type=int, default=None, help="stop after N sources")

    plan = sub.add_parser("plan", help="list stems that still need an image")
    plan.add_argument("--dataset", default=Path("dataset"), type=Path)
    plan.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
    plan.add_argument("--session", default=None, help="only this session, e.g. S01")
    plan.add_argument("--split", default=None, choices=["train", "val", "test"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            return _plan(args)
        return _ingest_command(args)
    except (IngestError, SplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _ingest_command(args: argparse.Namespace) -> int:
    mapping = args.map
    if mapping is None:
        candidate = args.incoming / "mapping.csv"
        if candidate.is_file():
            mapping = candidate

    if args.limit is not None and args.limit <= 0:
        raise IngestError("--limit must be positive")

    outcomes, warnings = ingest(
        incoming=args.incoming,
        dataset_dir=args.dataset,
        splits_path=args.splits,
        config_path=args.config,
        mapping_path=mapping,
        replace=args.replace,
        strict=args.strict,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")
    _print_outcomes(outcomes)
    counts = _summary(outcomes)
    print("summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 0 if counts.get("rejected", 0) == 0 else 1


def _plan(args: argparse.Namespace) -> int:
    rows = load_splits(args.splits)
    present = {
        path.stem
        for path in (args.dataset / "images").glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    } if (args.dataset / "images").is_dir() else set()

    selected = [
        row
        for row in rows
        if (args.session is None or row.session == args.session)
        and (args.split is None or row.split == args.split)
    ]
    if not selected:
        raise IngestError("no stems match the requested filters")

    missing = [row for row in selected if row.stem not in present]
    print(f"{'stem':10s} {'split':6s} {'session':8s} status")
    for row in selected:
        status = "present" if row.stem in present else "MISSING"
        print(f"{row.stem:10s} {row.split:6s} {row.session:8s} {status}")
    print(
        f"\n{len(selected) - len(missing)} of {len(selected)} present, {len(missing)} still needed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
