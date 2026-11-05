"""Train the U-Net + MobileNetV3-Small segmentation model on CPU.

The command refuses to run unless the frozen split has genuine annotated
images and masks in ``dataset/``. It never reads synthetic fixtures, and it
never reports a metric it did not measure: a missing checkpoint or an empty
split is an explicit unavailable result.

Determinism: the seed is recorded in ``run.json`` and applied to Python, NumPy
and PyTorch before any data is loaded. Augmentations are a pure function of
seed, epoch and sample index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.splits import DEFAULT_SPLITS, FROZEN_SHA256
from segmentation.data import (
    DatasetUnavailable,
    SegmentationDataset,
    SplitSelection,
    build_split,
)
from segmentation.losses import BCEDiceLoss
from segmentation.metrics import MetricResult, compute_metrics
from segmentation.model import (
    ARCHITECTURE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNetMobileNetV3Small,
    count_parameters,
)

CHECKPOINT_SCHEMA = "measureit.segmentation.checkpoint/1"


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    splits: str
    out: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    image_size: int
    workers: int
    pretrained: bool
    patience: int
    allow_partial: bool
    resume: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _loader(
    selection: SplitSelection,
    *,
    config: TrainConfig,
    train: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = SegmentationDataset(
        selection.samples,
        image_size=config.image_size,
        train=train,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.workers,
        generator=generator,
        persistent_workers=False,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, list[np.ndarray], list[np.ndarray], list[str]]:
    """Return (mean loss, probability maps, binary targets, stems)."""
    model.eval()
    total = 0.0
    count = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    stems: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        total += float(loss.item()) * images.size(0)
        count += images.size(0)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy()[:, 0])
        targets.extend(masks.cpu().numpy()[:, 0])
        stems.extend(str(stem) for stem in batch["stem"])
    return total / max(count, 1), probabilities, targets, stems


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, MetricResult]:
    loss, probabilities, targets, _ = collect_predictions(model, loader, loss_fn, device)
    return loss, compute_metrics(probabilities, targets)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: TrainConfig,
    epoch: int,
    metrics: MetricResult,
    best_iou: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": {
            "name": ARCHITECTURE,
            "in_channels": 3,
            "classes": 1,
            "pretrained": config.pretrained,
        },
        "image_size": config.image_size,
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "config": asdict(config),
        "epoch": int(epoch),
        "best_iou": float(best_iou),
        "val": metrics.as_dict(),
        "created_utc": utc_now(),
        "torch_version": str(torch.__version__),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> tuple[UNetMobileNetV3Small, dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{source} is not a {CHECKPOINT_SCHEMA} checkpoint")
    architecture = payload.get("architecture", {})
    model = UNetMobileNetV3Small(pretrained=False, classes=int(architecture.get("classes", 1)))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _write_run_json(
    path: Path,
    *,
    config: TrainConfig,
    train: SplitSelection,
    val: SplitSelection,
    parameters: int,
) -> None:
    payload = {
        "schema": "measureit.segmentation.run/1",
        "created_utc": utc_now(),
        "config": asdict(config),
        "split_sha256": FROZEN_SHA256,
        "train": {
            "expected": train.expected,
            "used": len(train.samples),
            "missing_images": len(train.missing_images),
            "missing_masks": len(train.missing_masks),
        },
        "val": {
            "expected": val.expected,
            "used": len(val.samples),
            "missing_images": len(val.missing_images),
            "missing_masks": len(val.missing_masks),
        },
        "model": {
            "name": ARCHITECTURE,
            "parameters": parameters,
            "image_size": config.image_size,
        },
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "torch_version": str(torch.__version__),
        "device": "cpu",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m segmentation.train",
        description="Train the segmentation model on the frozen, annotated dataset.",
    )
    parser.add_argument("--dataset", default=Path("dataset"), type=Path)
    parser.add_argument("--splits", default=DEFAULT_SPLITS, type=Path)
    parser.add_argument("--out", default=Path("runs/seg"), type=Path)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        print("ERROR: --epochs must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2

    config = TrainConfig(
        dataset=str(args.dataset),
        splits=str(args.splits),
        out=str(args.out),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        image_size=int(args.image_size),
        workers=int(args.workers),
        pretrained=bool(args.pretrained),
        patience=int(args.patience),
        allow_partial=bool(args.allow_partial),
        resume=None if args.resume is None else str(args.resume),
    )

    out = Path(config.out)
    last_path = out / "last.pt"
    best_path = out / "best.pt"
    if last_path.exists() and config.resume is None:
        print(
            f"ERROR: {out} already contains a run. Choose another --out or pass --resume.",
            file=sys.stderr,
        )
        return 2

    try:
        train_split = build_split(
            config.dataset, config.splits, "train", allow_partial=config.allow_partial
        )
        val_split = build_split(
            config.dataset, config.splits, "val", allow_partial=config.allow_partial
        )
    except DatasetUnavailable as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device("cpu")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = _loader(train_split, config=config, train=True, generator=generator)
    val_loader = _loader(val_split, config=config, train=False)
    train_loader.dataset.set_epoch(0)

    model = UNetMobileNetV3Small(pretrained=config.pretrained).to(device)
    loss_fn = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    _write_run_json(
        out / "run.json",
        config=config,
        train=train_split,
        val=val_split,
        parameters=count_parameters(model),
    )

    history_path = out / "history.csv"
    history_fields = [
        "epoch",
        "seconds",
        "lr",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
    ]
    write_header = not history_path.exists()

    start_epoch = 0
    best_iou = -1.0
    if config.resume is not None:
        try:
            model, payload = load_checkpoint(config.resume)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR: cannot resume: {exc}", file=sys.stderr)
            return 2
        model.to(device)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_iou = float(payload.get("best_iou", -1.0))
        print(f"resumed from {config.resume} at epoch {start_epoch}")

    if start_epoch >= config.epochs:
        print(f"nothing to do: epoch {start_epoch} >= {config.epochs}")
        return 0

    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        if write_header:
            writer.writeheader()
        stale = 0
        for epoch in range(start_epoch, config.epochs):
            started = time.perf_counter()
            train_loader.dataset.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, metrics = evaluate_model(model, val_loader, loss_fn, device)
            elapsed = time.perf_counter() - started
            scheduler.step(metrics.mean_iou if metrics.mean_iou is not None else 0.0)

            row = {
                "epoch": epoch,
                "seconds": f"{elapsed:.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_iou": "" if metrics.mean_iou is None else f"{metrics.mean_iou:.6f}",
                "val_precision": "" if metrics.precision is None else f"{metrics.precision:.6f}",
                "val_recall": "" if metrics.recall is None else f"{metrics.recall:.6f}",
                "val_map50": "" if metrics.map50 is None else f"{metrics.map50:.6f}",
                "val_map50_95": "" if metrics.map50_95 is None else f"{metrics.map50_95:.6f}",
            }
            writer.writerow(row)
            handle.flush()

            iou = metrics.mean_iou if metrics.mean_iou is not None else 0.0
            print(
                f"epoch {epoch + 1:03d}/{config.epochs}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  iou {iou:.4f}  {elapsed:.1f}s"
            )

            save_checkpoint(
                last_path,
                model=model,
                config=config,
                epoch=epoch,
                metrics=metrics,
                best_iou=max(best_iou, iou),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if iou > best_iou:
                best_iou = iou
                stale = 0
                save_checkpoint(
                    best_path,
                    model=model,
                    config=config,
                    epoch=epoch,
                    metrics=metrics,
                    best_iou=best_iou,
                )
                print(f"  saved best.pt (val iou {iou:.4f})")
            else:
                stale += 1
                if stale >= config.patience:
                    print(f"early stop: no improvement for {config.patience} epoch(s)")
                    break

    print(f"checkpoints: {last_path} and {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
