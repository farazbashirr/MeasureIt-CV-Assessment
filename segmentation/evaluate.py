"""Evaluate a trained checkpoint on a frozen split, or state that it is unavailable.

Metrics are only written when a checkpoint and genuine annotated images exist.
An empty split or a missing checkpoint produces an explicit
``available: false`` record with a reason, never a zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from dataset.splits import DEFAULT_SPLITS
from segmentation.data import (
    DatasetUnavailable,
    SegmentationDataset,
    build_split,
)
from segmentation.losses import BCEDiceLoss
from segmentation.metrics import binary_iou, compute_metrics
from segmentation.train import collect_predictions, load_checkpoint, utc_now


def _unavailable(out: Path, *, split: str, reason: str, checkpoint: str) -> None:
    payload = {
        "schema": "measureit.segmentation.evaluation/1",
        "created_utc": utc_now(),
        "checkpoint": checkpoint,
        "split": split,
        "available": False,
        "reason": reason,
        "expected": None,
        "evaluated": None,
        "complete": None,
        "missing_images": [],
        "missing_masks": [],
        "metrics": {
            "n_images": 0,
            "mean_iou": None,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_split(
    *,
    checkpoint: Path,
    dataset_dir: Path,
    splits_path: Path,
    split: str,
    out: Path,
    batch_size: int,
) -> dict:
    try:
        model, meta = load_checkpoint(checkpoint)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _unavailable(out, split=split, reason=str(exc), checkpoint=str(checkpoint))
        return {"available": False, "reason": str(exc)}

    try:
        selection = build_split(dataset_dir, splits_path, split, allow_partial=True)
    except DatasetUnavailable as exc:
        _unavailable(out, split=split, reason=str(exc), checkpoint=str(checkpoint))
        return {"available": False, "reason": str(exc)}

    image_size = int(meta.get("image_size", 512))
    normalization = meta.get("normalization", {})
    dataset = SegmentationDataset(
        selection.samples,
        image_size=image_size,
        train=False,
        mean=normalization.get("mean", (0.485, 0.456, 0.406)),
        std=normalization.get("std", (0.229, 0.224, 0.225)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = BCEDiceLoss()
    _, probabilities, targets, stems = collect_predictions(
        model, loader, loss_fn, torch.device("cpu")
    )
    metrics = compute_metrics(probabilities, targets)

    per_image: list[dict] = []
    for stem, probability, target in zip(stems, probabilities, targets):
        prediction = probability >= 0.5
        confidence = float(probability[prediction].mean()) if prediction.any() else 0.0
        per_image.append(
            {
                "stem": stem,
                "iou": f"{binary_iou(prediction, target > 0.5):.6f}",
                "confidence": f"{confidence:.6f}",
            }
        )
    per_image.sort(key=lambda row: row["stem"])

    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "iou", "confidence"])
        writer.writeheader()
        writer.writerows(per_image)

    payload = {
        "schema": "measureit.segmentation.evaluation/1",
        "created_utc": utc_now(),
        "checkpoint": str(checkpoint),
        "split": split,
        "available": True,
        "reason": None,
        "expected": selection.expected,
        "evaluated": len(selection.samples),
        "complete": selection.complete,
        "missing_images": list(selection.missing_images),
        "missing_masks": list(selection.missing_masks),
        "metrics": metrics.as_dict(),
        "model_epoch": meta.get("epoch"),
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m segmentation.evaluate",
        description="Evaluate a segmentation checkpoint on a frozen split.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=Path("dataset"), type=Path)
    parser.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=Path("runs/seg/eval"), type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2

    payload = evaluate_split(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset,
        splits_path=args.splits,
        split=args.split,
        out=args.out,
        batch_size=args.batch_size,
    )

    if not payload.get("available"):
        print(f"UNAVAILABLE: {payload['reason']}")
        print(f"metrics: {args.out / 'metrics.json'}")
        return 3

    metrics = payload["metrics"]
    print(
        f"split {args.split}: {payload['evaluated']}/{payload['expected']} annotated, "
        f"iou {metrics['mean_iou']:.4f}, precision {metrics['precision']:.4f}, "
        f"recall {metrics['recall']:.4f}, "
        f"mAP@0.5 {metrics['map50']:.4f}, mAP@0.5:0.95 {metrics['map50_95']:.4f}"
    )
    print(f"metrics: {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
