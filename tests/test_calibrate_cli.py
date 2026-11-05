import cv2
import numpy as np

from calibration.board import BoardSpec, generate, record_measurement
from calibration.calibrate import (
    EXIT_NO_IMAGES,
    EXIT_NO_USABLE_VIEWS,
    EXIT_NOT_MEASURED,
    main,
)


def _board(tmp_path):
    return generate(BoardSpec(), tmp_path / "board")


def test_unmeasured_board_blocks_calibration(tmp_path, capsys):
    paths = _board(tmp_path)
    code = main(
        [
            "--board",
            str(paths["spec"]),
            "--images",
            str(tmp_path / "images"),
            "--out",
            str(tmp_path / "camera.json"),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_NOT_MEASURED
    assert "BLOCKED" in captured.err
    assert "has not been measured" in captured.err


def test_empty_image_directory_is_reported(tmp_path, capsys):
    paths = _board(tmp_path)
    record_measurement(
        paths["spec"], square_x_mm=24.9, square_y_mm=24.9, instrument="test calliper"
    )
    images = tmp_path / "images"
    images.mkdir()

    code = main(
        [
            "--board",
            str(paths["spec"]),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "camera.json"),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_NO_IMAGES
    assert "contains no images" in captured.err


def test_blank_images_are_not_usable(tmp_path, capsys):
    paths = _board(tmp_path)
    record_measurement(
        paths["spec"], square_x_mm=24.9, square_y_mm=24.9, instrument="test calliper"
    )
    images = tmp_path / "images"
    images.mkdir()
    blank = np.full((1536, 2048), 255, dtype=np.uint8)
    for index in range(3):
        assert cv2.imwrite(str(images / f"blank_{index}.jpg"), blank)

    code = main(
        [
            "--board",
            str(paths["spec"]),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "camera.json"),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_NO_USABLE_VIEWS
    assert "usable views" in captured.err
    assert not (tmp_path / "camera.json").exists()
