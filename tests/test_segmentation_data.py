import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from segmentation.data import DatasetUnavailable, SegmentationDataset, build_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sandbox(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    shutil.copyfile(ROOT / "dataset" / "splits.csv", dataset / "splits.csv")
    return dataset


def _write_pair(dataset: Path, stem: str, width: int = 96, height: int = 72) -> None:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (30, 20), (60, 50), (220, 220, 220), -1)
    cv2.imwrite(str(dataset / "images" / f"{stem}.png"), image)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[20:50, 30:60] = 255
    cv2.imwrite(str(dataset / "masks" / f"{stem}.png"), mask)


def test_partial_split_is_refused_by_default(sandbox):
    _write_pair(sandbox, "S01_001")
    _write_pair(sandbox, "S01_002")

    with pytest.raises(DatasetUnavailable, match="2 of 70"):
        build_split(sandbox, sandbox / "splits.csv", "train")

    selection = build_split(
        sandbox, sandbox / "splits.csv", "train", allow_partial=True
    )
    assert selection.expected == 70
    assert len(selection.samples) == 2
    assert len(selection.missing_masks) == 68
    assert not selection.complete


def test_empty_split_is_unavailable(sandbox):
    with pytest.raises(DatasetUnavailable, match="0 of 70"):
        build_split(sandbox, sandbox / "splits.csv", "train", allow_partial=True)


def test_unknown_split_is_rejected(sandbox):
    with pytest.raises(ValueError):
        build_split(sandbox, sandbox / "splits.csv", "holdout")


def test_dataset_item_shapes_and_binary_mask(sandbox):
    _write_pair(sandbox, "S01_001")
    selection = build_split(sandbox, sandbox / "splits.csv", "train", allow_partial=True)
    dataset = SegmentationDataset(selection.samples, image_size=64)

    item = dataset[0]
    assert item["stem"] == "S01_001"
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})
    assert torch.isfinite(item["image"]).all()


def test_augmentation_is_deterministic_for_same_epoch(sandbox):
    _write_pair(sandbox, "S01_003")
    selection = build_split(sandbox, sandbox / "splits.csv", "train", allow_partial=True)

    first = SegmentationDataset(selection.samples, image_size=64, train=True, seed=5)
    second = SegmentationDataset(selection.samples, image_size=64, train=True, seed=5)
    first.set_epoch(1)
    second.set_epoch(1)
    assert torch.equal(first[0]["image"], second[0]["image"])
    assert torch.equal(first[0]["mask"], second[0]["mask"])

    second.set_epoch(2)
    assert not torch.equal(first[0]["image"], second[0]["image"])


def test_non_binary_mask_is_rejected(sandbox):
    _write_pair(sandbox, "S01_004")
    mask_path = sandbox / "masks" / "S01_004.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask[25, 35] = 128
    cv2.imwrite(str(mask_path), mask)

    selection = build_split(sandbox, sandbox / "splits.csv", "train", allow_partial=True)
    dataset = SegmentationDataset(selection.samples, image_size=64)
    with pytest.raises(DatasetUnavailable, match="not binary"):
        _ = dataset[0]


def test_mask_image_size_mismatch_is_rejected(sandbox):
    _write_pair(sandbox, "S01_005")
    cv2.imwrite(
        str(sandbox / "masks" / "S01_005.png"),
        np.zeros((32, 32), dtype=np.uint8),
    )
    selection = build_split(sandbox, sandbox / "splits.csv", "train", allow_partial=True)
    dataset = SegmentationDataset(selection.samples, image_size=64)
    with pytest.raises(DatasetUnavailable, match="differ"):
        _ = dataset[0]
