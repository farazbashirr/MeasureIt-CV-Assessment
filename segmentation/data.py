"""Load annotated images from the frozen train/validation/test split.

The loader never invents a mask or a split. It reads ``dataset/splits.csv``
(SHA-verified), ``dataset/images/`` and ``dataset/masks/``, and raises
``DatasetUnavailable`` when the requested split has no genuine annotated
images. Partial splits are refused unless the caller explicitly opts in, and
the counts are always reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.splits import DEFAULT_SPLITS, image_for_stem, load_splits
from measureit.io import load_image
from segmentation.model import IMAGENET_MEAN, IMAGENET_STD


class DatasetUnavailable(RuntimeError):
    """The requested split has no usable real images and masks."""


@dataclass(frozen=True)
class Sample:
    stem: str
    split: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class SplitSelection:
    split: str
    expected: int
    samples: tuple[Sample, ...]
    missing_images: tuple[str, ...]
    missing_masks: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_images and not self.missing_masks


def build_split(
    dataset_dir: str | Path,
    splits_path: str | Path = DEFAULT_SPLITS,
    split: str = "train",
    *,
    allow_partial: bool = False,
) -> SplitSelection:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")

    root = Path(dataset_dir)
    rows = [row for row in load_splits(splits_path) if row.split == split]
    samples: list[Sample] = []
    missing_images: list[str] = []
    missing_masks: list[str] = []

    for row in rows:
        image_path = image_for_stem(root / "images", row.stem)
        mask_path = root / "masks" / f"{row.stem}.png"
        if image_path is None:
            missing_images.append(row.stem)
        if not mask_path.is_file():
            missing_masks.append(row.stem)
        if image_path is not None and mask_path.is_file():
            samples.append(
                Sample(
                    stem=row.stem,
                    split=row.split,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

    if not samples:
        raise DatasetUnavailable(
            f"split {split!r}: 0 of {len(rows)} stems have both an image and a mask. "
            "Genuine annotated images are required; synthetic fixtures are not accepted."
        )

    if (missing_images or missing_masks) and not allow_partial:
        raise DatasetUnavailable(
            f"split {split!r}: {len(samples)} of {len(rows)} stems complete; "
            f"{len(missing_images)} missing image(s), {len(missing_masks)} missing mask(s). "
            "Finish annotating, or pass --allow-partial and report the reduced count."
        )

    return SplitSelection(
        split=split,
        expected=len(rows),
        samples=tuple(samples),
        missing_images=tuple(missing_images),
        missing_masks=tuple(missing_masks),
    )


class SegmentationDataset(Dataset):
    """Image/mask pairs resized to a square model input.

    Augmentations (flip, small rotation, brightness/contrast) are deterministic
    for a given ``seed``, ``epoch`` and sample index, so a run is reproducible.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        image_size: int = 512,
        train: bool = False,
        seed: int = 0,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, mask: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        if rng.random() < 0.2:
            image = image[::-1]
            mask = mask[::-1]

        if rng.random() < 0.5:
            angle = float(rng.uniform(-10.0, 10.0))
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                np.ascontiguousarray(image),
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                np.ascontiguousarray(mask),
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )

        if rng.random() < 0.5:
            gain = float(rng.uniform(0.9, 1.1))
            bias = float(rng.uniform(-10.0, 10.0))
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = load_image(sample.image_path)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise DatasetUnavailable(f"{sample.stem}: cannot read mask {sample.mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            raise DatasetUnavailable(
                f"{sample.stem}: image {image.shape[1]}x{image.shape[0]} and mask "
                f"{mask.shape[1]}x{mask.shape[0]} differ"
            )
        values = {int(value) for value in np.unique(mask)}
        if not values.issubset({0, 255}):
            raise DatasetUnavailable(f"{sample.stem}: mask is not binary: {sorted(values)}")
        binary = (mask > 0).astype(np.uint8)
        if not binary.any():
            raise DatasetUnavailable(f"{sample.stem}: mask is empty")

        if self.train:
            rng = np.random.default_rng(
                (self.seed + index + self.epoch * 1_000_003) % (2**32)
            )
            image, binary = self._augment(image, binary, rng)

        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(binary[None].astype(np.float32))
        )
        return {"image": image_tensor, "mask": mask_tensor, "stem": sample.stem}
