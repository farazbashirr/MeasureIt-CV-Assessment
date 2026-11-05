import cv2
import numpy as np
import pytest

from dataset.quality import (
    QualityConfig,
    QualityError,
    assess,
    canonical_suffix,
    dhash64,
    format_flags,
    hamming,
    parse_flags,
)


def _checkerboard(width=1024, height=768, cell=32) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((xx // cell + yy // cell) % 2).astype(np.uint8)
    values = 90 + pattern * 80
    return np.dstack([values, values, values])


def test_sharp_image_outranks_blurred_image():
    config = QualityConfig.load()
    sharp = assess(_checkerboard(), config)
    blurred = assess(cv2.GaussianBlur(_checkerboard(), (0, 0), 6.0), config)

    assert sharp.blur_variance > blurred.blur_variance
    assert "blurry" not in sharp.flags
    assert "blurry" in blurred.flags


def test_exposure_flags():
    config = QualityConfig.load()
    black = assess(np.zeros((768, 1024, 3), dtype=np.uint8), config)
    white = assess(np.full((768, 1024, 3), 255, dtype=np.uint8), config)
    middle = assess(np.full((768, 1024, 3), 128, dtype=np.uint8), config)

    assert "underexposed" in black.flags
    assert "overexposed" in white.flags
    assert "underexposed" not in middle.flags
    assert "overexposed" not in middle.flags


def test_dhash_is_stable_and_sensitive():
    image = _checkerboard()
    shifted = np.roll(image, 8, axis=1)
    assert dhash64(image) == dhash64(image)
    assert hamming(dhash64(image), dhash64(image)) == 0
    assert hamming(dhash64(image), dhash64(shifted)) > 0


def test_flag_round_trip():
    flags = ("blurry", "underexposed")
    assert parse_flags(format_flags(flags)) == flags
    assert parse_flags("") == ()


def test_config_rejects_unknown_schema(tmp_path):
    path = tmp_path / "quality.json"
    path.write_text('{"schema": "other/1"}', encoding="utf-8")
    with pytest.raises(QualityError, match="unsupported quality schema"):
        QualityConfig.load(path)


def test_config_rejects_missing_key(tmp_path):
    path = tmp_path / "quality.json"
    path.write_text('{"schema": "measureit.quality/1"}', encoding="utf-8")
    with pytest.raises(QualityError, match="missing or has an invalid key"):
        QualityConfig.load(path)


def test_canonical_suffix():
    assert canonical_suffix("a.JPEG") == ".jpg"
    assert canonical_suffix("a.JPG") == ".jpg"
    assert canonical_suffix("a.png") == ".png"
    with pytest.raises(QualityError):
        canonical_suffix("a.tif")
