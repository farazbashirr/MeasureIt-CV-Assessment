import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from dataset.annotate import main as annotate_main
from dataset.annotations import (
    ObjectPolygon,
    PolygonAnnotation,
    load_polygon,
    mask_iou,
    mask_to_polygon,
    polygon_area,
    polygon_to_mask,
    validate_mask,
    validate_polygon,
    write_polygon,
)
from dataset.quality import QualityConfig
from dataset.splits import load_splits
from dataset.state import collect_state

ROOT = Path(__file__).resolve().parents[1]
SIZE = (2048, 1536)
RECT = ((600.0, 500.0), (1100.0, 500.0), (1100.0, 900.0), (600.0, 900.0))


@pytest.fixture()
def sandbox(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "config").mkdir(parents=True)
    (dataset / "images").mkdir()
    (dataset / "masks").mkdir()
    (dataset / "annotations").mkdir()
    shutil.copyfile(ROOT / "dataset" / "splits.csv", dataset / "splits.csv")
    shutil.copyfile(
        ROOT / "dataset" / "config" / "quality.json",
        dataset / "config" / "quality.json",
    )
    return dataset


def _write_photo(dataset: Path, stem: str) -> Path:
    image = np.full((SIZE[1], SIZE[0], 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (600, 500), (1100, 900), (200, 200, 200), -1)
    path = dataset / "images" / f"{stem}.jpg"
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, quality=95)
    return path


def _annotation(stem: str = "S01_001", points=RECT) -> PolygonAnnotation:
    return PolygonAnnotation(
        stem=stem,
        image=f"{stem}.jpg",
        image_size=SIZE,
        objects=(ObjectPolygon(label="object", points=tuple(points)),),
    )


def _state(dataset: Path):
    rows = load_splits(dataset / "splits.csv")
    config = QualityConfig.load(dataset / "config" / "quality.json")
    return collect_state(dataset, rows, config)


def _cli(dataset: Path, *extra: str) -> int:
    return annotate_main(
        [
            *extra,
            "--dataset",
            str(dataset),
            "--splits",
            str(dataset / "splits.csv"),
            "--config",
            str(dataset / "config" / "quality.json"),
        ]
    )


def test_polygon_area_and_mask_agree():
    mask = polygon_to_mask(RECT, SIZE)
    assert mask.shape == (SIZE[1], SIZE[0])
    assert set(np.unique(mask)).issubset({0, 255})
    assert polygon_area(RECT) == pytest.approx(200_000.0)

    filled = float(np.count_nonzero(mask))
    assert filled == pytest.approx(polygon_area(RECT), rel=0.01)
    assert mask_iou(mask, mask) == 1.0


def test_mask_round_trip_is_stable():
    mask = polygon_to_mask(RECT, SIZE)
    recovered = mask_to_polygon(mask, epsilon_px=1.0)
    rebuilt = polygon_to_mask(recovered, SIZE)

    assert len(recovered) >= 3
    assert mask_iou(mask, rebuilt) > 0.98


def test_validate_polygon_accepts_rectangle():
    assert validate_polygon(_annotation(), expected_size=SIZE, expected_stem="S01_001") == []


def test_validate_polygon_rejects_bad_geometry():
    outside = _annotation(points=((600, 500), (3000, 500), (1100, 900)))
    too_few = _annotation(points=((600, 500), (1100, 900)))
    bowtie = _annotation(points=((200, 200), (1200, 1200), (1200, 300), (300, 1200)))

    assert any("x outside" in problem for problem in validate_polygon(outside, expected_size=SIZE))
    assert any("at least 3" in problem for problem in validate_polygon(too_few, expected_size=SIZE))
    assert any("self-intersecting" in problem for problem in validate_polygon(bowtie, expected_size=SIZE))
    assert any(
        "stem" in problem
        for problem in validate_polygon(_annotation(), expected_size=SIZE, expected_stem="S01_002")
    )


def test_validate_mask_rejects_bad_masks():
    config = QualityConfig.load(ROOT / "dataset" / "config" / "quality.json")

    good = polygon_to_mask(RECT, SIZE)
    assert validate_mask(good, expected_size=SIZE, config=config) == []

    wrong_size = np.zeros((768, 1024), dtype=np.uint8)
    assert validate_mask(wrong_size, expected_size=SIZE, config=config)

    non_binary = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    non_binary[500:900, 600:1100] = 128
    assert validate_mask(non_binary, expected_size=SIZE, config=config)

    empty = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    assert validate_mask(empty, expected_size=SIZE, config=config)

    two_parts = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    two_parts[100:200, 100:200] = 255
    two_parts[400:500, 400:500] = 255
    assert any("connected components" in problem for problem in validate_mask(two_parts, expected_size=SIZE, config=config))


def test_annotation_without_image_is_a_problem(sandbox):
    write_polygon(_annotation(), sandbox / "annotations" / "S01_001.json")

    state = _state(sandbox)
    problems = state.problems
    assert any("without an image" in problem for problem in problems)
    assert _cli(sandbox, "validate") == 1


def test_unknown_annotation_is_reported(sandbox):
    write_polygon(_annotation(stem="S99_001"), sandbox / "annotations" / "S99_001.json")

    state = _state(sandbox)
    assert any("S99_001" in message for message in state.unknown)


def test_cli_end_to_end(sandbox, capsys):
    _write_photo(sandbox, "S01_001")
    annotation_path = sandbox / "annotations" / "S01_001.json"
    write_polygon(_annotation(), annotation_path)

    assert _cli(sandbox, "validate") == 0
    assert _cli(sandbox, "polygons-to-masks") == 0
    mask_path = sandbox / "masks" / "S01_001.png"
    assert mask_path.is_file()
    assert set(np.unique(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))).issubset({0, 255})

    assert _cli(sandbox, "validate") == 0

    out = sandbox / "review"
    assert _cli(sandbox, "review", "--out", str(out)) == 0
    assert (out / "S01_001_overlay.png").is_file()
    with (out / "summary.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["stem"] == "S01_001"
    assert float(rows[0]["iou"]) > 0.98

    assert _cli(sandbox, "masks-to-polygons", "--force") == 0
    recovered = load_polygon(annotation_path)
    assert len(recovered.objects[0].points) >= 3
    assert recovered.image_size == SIZE


def test_load_polygon_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "other/1"}), encoding="utf-8")
    with pytest.raises(Exception, match="unsupported schema"):
        load_polygon(path)


def test_mask_iou_of_disjoint_masks_is_zero():
    left = np.zeros((100, 100), dtype=np.uint8)
    right = np.zeros((100, 100), dtype=np.uint8)
    left[10:20, 10:20] = 255
    right[50:60, 50:60] = 255
    assert mask_iou(left, right) == 0.0
