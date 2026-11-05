import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from dataset.ingest import main as ingest_main
from dataset.splits import FROZEN_SHA256, file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sandbox(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "config").mkdir(parents=True)
    shutil.copyfile(ROOT / "dataset" / "splits.csv", dataset / "splits.csv")
    shutil.copyfile(
        ROOT / "dataset" / "config" / "quality.json",
        dataset / "config" / "quality.json",
    )
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    return dataset, incoming


def _args(dataset: Path, incoming: Path, *extra: str) -> list[str]:
    return [
        "ingest",
        "--incoming",
        str(incoming),
        "--dataset",
        str(dataset),
        "--splits",
        str(dataset / "splits.csv"),
        "--config",
        str(dataset / "config" / "quality.json"),
        *extra,
    ]


def _write_photo(
    path: Path,
    width: int = 2560,
    height: int = 1920,
    cell: int = 64,
    low: int = 90,
    high: int = 170,
    orientation: int | None = None,
) -> None:
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((xx // cell + yy // cell) % 2).astype(np.uint8)
    values = low + pattern * (high - low)
    rgb = np.dstack([values, values, values])
    image = Image.fromarray(rgb)
    if orientation is None:
        image.save(path, quality=95)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, quality=95, exif=exif)


def _uniform_photo(path: Path, width: int = 2048, height: int = 1536) -> None:
    Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8)).save(path, quality=95)


def _log_rows(dataset: Path) -> list[dict[str, str]]:
    log = dataset / "capture_log.csv"
    with log.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_ingest_normalises_archives_and_logs(sandbox, capsys):
    dataset, incoming = sandbox
    source = incoming / "S01_001.jpg"
    _write_photo(source)

    code = ingest_main(_args(dataset, incoming))

    assert code == 0
    processed = dataset / "images" / "S01_001.jpg"
    assert processed.is_file()
    image = cv2.imread(str(processed))
    assert image.shape == (1536, 2048, 3)
    assert (dataset / "originals" / "S01_001.jpg").read_bytes() == source.read_bytes()

    rows = _log_rows(dataset)
    assert len(rows) == 1
    assert rows[0]["stem"] == "S01_001"
    assert rows[0]["split"] == "train"
    assert rows[0]["original_width"] == "2560"
    assert rows[0]["width"] == "2048"
    assert rows[0]["height"] == "1536"
    assert rows[0]["original_sha256"] != rows[0]["processed_sha256"]
    assert file_sha256(dataset / "splits.csv") == FROZEN_SHA256
    assert "ingested" in capsys.readouterr().out


def test_exif_orientation_is_honoured(sandbox):
    dataset, incoming = sandbox
    _write_photo(incoming / "S01_002.jpg", width=1920, height=2560, orientation=6)

    assert ingest_main(_args(dataset, incoming)) == 0
    image = cv2.imread(str(dataset / "images" / "S01_002.jpg"))
    assert image.shape == (1536, 2048, 3)
    assert _log_rows(dataset)[0]["height"] == "1536"


def test_mapping_file_assigns_arbitrary_names(sandbox):
    dataset, incoming = sandbox
    _write_photo(incoming / "IMG_1042.jpeg")
    (incoming / "mapping.csv").write_text(
        "source,stem\nIMG_1042.jpeg,S01_003\n", encoding="utf-8"
    )

    assert ingest_main(_args(dataset, incoming)) == 0
    assert (dataset / "images" / "S01_003.jpg").is_file()
    assert not (dataset / "images" / "IMG_1042.jpg").exists()


def test_wrong_aspect_is_rejected(sandbox):
    dataset, incoming = sandbox
    _write_photo(incoming / "S01_004.jpg", width=2000, height=2000)

    assert ingest_main(_args(dataset, incoming)) == 1
    assert not (dataset / "images" / "S01_004.jpg").exists()
    assert not (dataset / "capture_log.csv").exists()


def test_corrupted_file_is_rejected(sandbox):
    dataset, incoming = sandbox
    (incoming / "S01_005.jpg").write_bytes(b"this is not a jpeg")

    assert ingest_main(_args(dataset, incoming)) == 1
    assert not (dataset / "images" / "S01_005.jpg").exists()


def test_unknown_stem_is_refused(sandbox):
    dataset, incoming = sandbox
    _write_photo(incoming / "S99_001.jpg")

    assert ingest_main(_args(dataset, incoming)) == 2
    assert not list((dataset / "images").glob("*.jpg"))


def test_duplicate_content_is_rejected(sandbox, capsys):
    dataset, incoming = sandbox
    _write_photo(incoming / "S01_006.jpg")
    shutil.copyfile(incoming / "S01_006.jpg", incoming / "S01_007.jpg")

    assert ingest_main(_args(dataset, incoming)) == 1
    assert (dataset / "images" / "S01_006.jpg").is_file()
    assert not (dataset / "images" / "S01_007.jpg").exists()
    assert "identical processed content" in capsys.readouterr().out


def test_mapping_with_duplicate_stem_is_refused(sandbox):
    dataset, incoming = sandbox
    _write_photo(incoming / "a.jpg")
    _write_photo(incoming / "b.jpg", low=150, high=200)
    (incoming / "mapping.csv").write_text(
        "source,stem\na.jpg,S01_008\nb.jpg,S01_008\n", encoding="utf-8"
    )

    assert ingest_main(_args(dataset, incoming)) == 2


def test_dry_run_writes_nothing(sandbox, capsys):
    dataset, incoming = sandbox
    _write_photo(incoming / "S01_009.jpg")

    assert ingest_main(_args(dataset, incoming, "--dry-run")) == 0
    assert not (dataset / "images").exists()
    assert not (dataset / "originals").exists()
    assert not (dataset / "capture_log.csv").exists()
    assert "would-ingest" in capsys.readouterr().out


def test_second_run_skips_existing(sandbox, capsys):
    dataset, incoming = sandbox
    _write_photo(incoming / "S01_010.jpg")

    assert ingest_main(_args(dataset, incoming)) == 0
    assert ingest_main(_args(dataset, incoming)) == 0
    assert "skipped" in capsys.readouterr().out
    assert len(_log_rows(dataset)) == 1


def test_blur_is_flagged_but_ingested(sandbox):
    dataset, incoming = sandbox
    _uniform_photo(incoming / "S02_001.jpg")

    assert ingest_main(_args(dataset, incoming)) == 0
    assert "blurry" in _log_rows(dataset)[0]["flags"]
    assert (dataset / "images" / "S02_001.jpg").is_file()


def test_strict_mode_rejects_flagged_frames(sandbox):
    dataset, incoming = sandbox
    _uniform_photo(incoming / "S02_002.jpg")

    assert ingest_main(_args(dataset, incoming, "--strict")) == 1
    assert not (dataset / "images" / "S02_002.jpg").exists()
