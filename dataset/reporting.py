"""Generate the dataset report from the files that actually exist.

The report is descriptive, never aspirational: expected counts come from the
frozen split, present counts come from disk, and anything missing is called
missing. Synthetic fixtures from the test suite never appear here because the
tests write only to temporary directories.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from dataset.quality import QualityConfig, hamming
from dataset.splits import DEFAULT_SPLITS, FROZEN_SHA256, load_splits
from dataset.state import MASK_POLYGON_IOU_MIN, DatasetState, collect_state


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_inventory(path: Path, state: DatasetState) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stem",
                "split",
                "session",
                "condition",
                "image",
                "width",
                "height",
                "blur_laplacian_var",
                "mean_luma",
                "flags",
                "mask",
                "polygon",
            ]
        )
        for item in state.stems:
            quality = item.quality
            writer.writerow(
                [
                    item.stem,
                    item.split,
                    item.session,
                    item.condition,
                    "" if item.image is None else item.image.name,
                    "" if item.image_size is None else item.image_size[0],
                    "" if item.image_size is None else item.image_size[1],
                    "" if quality is None else f"{quality.blur_variance:.2f}",
                    "" if quality is None else f"{quality.mean_luma:.2f}",
                    "" if quality is None else ";".join(quality.flags),
                    int(item.mask_path is not None),
                    int(item.polygon_path is not None),
                ]
            )


def _write_quality(path: Path, state: DatasetState) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stem",
                "blur_laplacian_var",
                "mean_luma",
                "dark_pct",
                "bright_pct",
                "flags",
            ]
        )
        for item in state.present:
            quality = item.quality
            if quality is None:
                continue
            writer.writerow(
                [
                    item.stem,
                    f"{quality.blur_variance:.2f}",
                    f"{quality.mean_luma:.2f}",
                    f"{quality.dark_pct:.2f}",
                    f"{quality.bright_pct:.2f}",
                    ";".join(quality.flags),
                ]
            )


def _duplicate_pairs(state: DatasetState, threshold: int) -> list[tuple[str, str, int]]:
    present = [item for item in state.present if item.dhash is not None]
    pairs: list[tuple[str, str, int]] = []
    for index, left in enumerate(present):
        for right in present[index + 1 :]:
            distance = hamming(left.dhash, right.dhash)
            if distance <= threshold:
                pairs.append((left.stem, right.stem, distance))
    pairs.sort(key=lambda item: item[2])
    return pairs


def _flag_counts(state: DatasetState) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in state.present:
        if item.quality is not None:
            for flag in item.quality.flags:
                counts[flag] += 1
    return counts


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(
    *,
    dataset_dir: Path,
    splits_path: Path,
    config_path: Path,
    out_dir: Path,
) -> dict[str, int]:
    rows = load_splits(splits_path)
    config = QualityConfig.load(config_path)
    state = collect_state(dataset_dir, rows, config)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_inventory(out_dir / "inventory.csv", state)
    _write_quality(out_dir / "quality.csv", state)

    present = state.present
    masks = [item for item in present if item.has_mask]
    polygons = [item for item in present if item.polygon_path is not None]
    both = [item for item in present if item.mask_path and item.polygon_path]
    unannotated = [item for item in present if not item.has_annotation]
    missing_masks = len(present) - len(masks)
    duplicates = _duplicate_pairs(state, config.hamming_max)

    expected_by_split = Counter(row.split for row in rows)
    present_by_split = Counter(item.split for item in present)
    expected_by_session = Counter(row.session for row in rows)
    present_by_session = Counter(item.session for item in present)

    resolutions = Counter(
        f"{item.image_size[0]}x{item.image_size[1]}"
        for item in present
        if item.image_size is not None
    )
    flags = _flag_counts(state)

    blur_values = [
        item.quality.blur_variance for item in present if item.quality is not None
    ]

    lines: list[str] = []
    lines.append("# Dataset report\n")
    lines.append(f"- Generated: {_utc_now()}")
    lines.append(f"- Split manifest: `{splits_path.as_posix()}`")
    lines.append(f"- Manifest SHA-256: `{FROZEN_SHA256}` (verified, unchanged)")
    lines.append(
        f"- Working resolution: {config.working_size[0]}x{config.working_size[1]}"
    )
    lines.append(f"- Images present: **{len(present)} / {len(rows)}**")
    lines.append(
        f"- Masks: **{len(masks)}**; polygon annotations: **{len(polygons)}**; "
        f"images with both: **{len(both)}**"
    )
    lines.append(f"- Images with no annotation: **{len(unannotated)}**")
    lines.append(f"- Images still missing a mask: **{missing_masks}**")
    lines.append(
        f"- Files outside the frozen split: **{len(state.unknown)}**"
    )
    lines.append("")

    lines.append("## Provenance\n")
    lines.append(
        "Every image counted here is a file under `dataset/images/`. This repository "
        "does not generate dataset images or masks; the test suite writes synthetic "
        "fixtures only to temporary directories, never into `dataset/`. Originals are "
        "archived in `dataset/originals/` (gitignored) and hashed in "
        "`dataset/capture_log.csv`."
    )
    lines.append("")

    lines.append("## By split\n")
    lines.extend(
        _table(
            ["split", "expected", "present", "missing"],
            [
                [
                    split,
                    str(expected_by_split.get(split, 0)),
                    str(present_by_split.get(split, 0)),
                    str(expected_by_split.get(split, 0) - present_by_split.get(split, 0)),
                ]
                for split in ("train", "val", "test")
            ],
        )
    )
    lines.append("")

    lines.append("## By session\n")
    lines.extend(
        _table(
            ["session", "split", "expected", "present", "missing"],
            [
                [
                    session,
                    next(row.split for row in rows if row.session == session),
                    str(expected_by_session[session]),
                    str(present_by_session.get(session, 0)),
                    str(expected_by_session[session] - present_by_session.get(session, 0)),
                ]
                for session in sorted(expected_by_session)
            ],
        )
    )
    lines.append("")

    lines.append("## Resolution\n")
    if resolutions:
        lines.extend(_table(["size", "images"], [[size, str(count)] for size, count in resolutions.most_common()]))
    else:
        lines.append("No images yet.")
    lines.append("")

    lines.append("## Quality\n")
    if present:
        lines.append(
            f"- Laplacian variance: mean {sum(blur_values) / len(blur_values):.1f}, "
            f"min {min(blur_values):.1f}"
        )
        if flags:
            lines.append(
                "- Flagged: "
                + ", ".join(f"{name} {count}" for name, count in sorted(flags.items()))
            )
        else:
            lines.append("- Flagged: none")
        lines.append(
            "- Thresholds: `dataset/config/quality.json` "
            f"(blur < {config.blur_min}, mean luma outside "
            f"{config.mean_min}..{config.mean_max})"
        )
    else:
        lines.append("No images yet.")
    lines.append("")

    lines.append("## Duplicate candidates\n")
    if duplicates:
        lines.extend(
            _table(
                ["image A", "image B", "dHash distance"],
                [[left, right, str(distance)] for left, right, distance in duplicates],
            )
        )
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Annotation problems\n")
    problems = state.problems
    if problems:
        for problem in problems[:20]:
            lines.append(f"- {problem}")
        if len(problems) > 20:
            lines.append(f"- ... and {len(problems) - 20} more (see inventory.csv)")
    else:
        lines.append("None.")
    lines.append("")

    pending = [item for item in state.stems if item.image is None]
    lines.append("## Pending stems\n")
    lines.append(
        f"{len(pending)} of {len(rows)} stems have no image yet. "
        "`python -m dataset.ingest plan` prints them; `inventory.csv` lists every stem."
    )
    lines.append("")
    lines.append(
        f"_Generated by `dataset.reporting`; mask/polygon agreement gate is IoU >= "
        f"{MASK_POLYGON_IOU_MIN:.2f}._"
    )

    (out_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "expected": len(rows),
        "present": len(present),
        "masks": len(masks),
        "polygons": len(polygons),
        "unannotated": len(unannotated),
        "duplicates": len(duplicates),
        "problems": len(problems),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.reporting",
        description="Write an honest dataset report from files on disk.",
    )
    parser.add_argument("--dataset", default=Path("dataset"), type=Path)
    parser.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
    parser.add_argument("--config", default=Path("dataset/config/quality.json"), type=Path)
    parser.add_argument("--out", default=Path("dataset/report"), type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_report(
        dataset_dir=args.dataset,
        splits_path=args.splits,
        config_path=args.config,
        out_dir=args.out,
    )
    print(
        "dataset: {present}/{expected} images, {masks} masks, {polygons} polygons, "
        "{duplicates} duplicate pair(s), {problems} problem(s)".format(**summary)
    )
    print(f"report: {args.out / 'dataset_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
