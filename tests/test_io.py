from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from measureit.io import ImageLoadError, load_image


def _image(path: Path, size=(800, 600), orientation=None) -> Path:
    image = Image.new("RGB", size, (255, 255, 255))
    if orientation is None:
        image.save(path)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)
    return path


def test_resizes_to_the_working_frame(tmp_path):
    path = _image(tmp_path / "landscape.jpg")
    loaded = load_image(path, target_size=(400, 300))
    assert loaded.shape == (300, 400, 3)
    assert loaded.dtype == np.uint8


def test_keeps_source_size_when_no_target(tmp_path):
    path = _image(tmp_path / "native.jpg", size=(640, 480))
    assert load_image(path).shape == (480, 640, 3)


def test_exif_orientation_is_applied(tmp_path):
    path = _image(tmp_path / "rotated.jpg", orientation=6)
    loaded = load_image(path, target_size=(300, 400))
    assert loaded.shape == (400, 300, 3)


def test_wrong_orientation_is_refused_not_stretched(tmp_path):
    path = _image(tmp_path / "portrait.jpg", orientation=6)
    with pytest.raises(ImageLoadError, match="does not match"):
        load_image(path, target_size=(400, 300))


def test_square_target_refuses_a_landscape_source(tmp_path):
    path = _image(tmp_path / "landscape.jpg")
    with pytest.raises(ImageLoadError, match="does not match"):
        load_image(path, target_size=(400, 400))


def test_unreadable_file_is_reported(tmp_path):
    path = tmp_path / "not_an_image.jpg"
    path.write_bytes(b"not a jpeg")
    with pytest.raises(ImageLoadError, match="cannot read"):
        load_image(path)
