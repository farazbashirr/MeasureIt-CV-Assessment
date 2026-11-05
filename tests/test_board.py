import json

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

from calibration.board import (
    BoardError,
    BoardSpec,
    MissingMeasurementError,
    build_charuco,
    generate,
    load_spec,
    record_measurement,
)

MM_PER_POINT = 25.4 / 72.0


def test_generate_writes_board_pdf_and_spec(tmp_path):
    paths = generate(BoardSpec(), tmp_path)

    assert paths["pdf"].is_file()
    assert paths["png"].is_file()
    assert paths["spec"].is_file()

    reader = PdfReader(str(paths["pdf"]))
    assert len(reader.pages) == 2
    box = reader.pages[0].mediabox
    assert float(box.width) * MM_PER_POINT == pytest.approx(297.0, abs=0.5)
    assert float(box.height) * MM_PER_POINT == pytest.approx(210.0, abs=0.5)

    image = cv2.imread(str(paths["png"]), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (7 * 300, 9 * 300)
    assert set(np.unique(image)).issubset({0, 255})

    data = json.loads(paths["spec"].read_text(encoding="utf-8"))
    assert data["inner_corners"] == 48
    assert data["measured_square_x_mm"] is None
    assert "not the paper" in data["notes"]


def test_unmeasured_board_blocks_calibration():
    spec = BoardSpec()
    with pytest.raises(MissingMeasurementError):
        _ = spec.measured_square_mm
    with pytest.raises(MissingMeasurementError):
        build_charuco(spec, measured=True)


def test_measurement_round_trip(tmp_path):
    paths = generate(BoardSpec(), tmp_path)

    spec, warnings = record_measurement(
        paths["spec"],
        square_x_mm=24.87,
        square_y_mm=24.90,
        instrument="test calliper, 0.01 mm, +/-0.02 mm",
    )
    assert warnings == []
    assert spec.measured_square_mm == pytest.approx(24.885, abs=1e-9)

    reloaded = load_spec(paths["spec"])
    assert reloaded.measured_square_mm == pytest.approx(24.885, abs=1e-9)
    assert reloaded.measurement_instrument.startswith("test calliper")
    assert reloaded.axis_difference_pct == pytest.approx(0.1205, abs=0.01)


def test_stretched_print_is_warned(tmp_path):
    paths = generate(BoardSpec(), tmp_path)
    _, warnings = record_measurement(
        paths["spec"], square_x_mm=25.0, square_y_mm=25.6, instrument="test rule"
    )
    assert any("stretched" in warning for warning in warnings)


def test_marker_must_be_smaller_than_square():
    with pytest.raises(BoardError):
        BoardSpec(marker_mm_nominal=25.0).validate()


def test_board_must_fit_the_page():
    with pytest.raises(BoardError):
        BoardSpec(squares_x=12, squares_y=12, square_mm_nominal=30.0).validate()


def test_measured_scale_drives_geometry():
    nominal = build_charuco(BoardSpec(), measured=False).getChessboardCorners()
    measured = build_charuco(
        BoardSpec(measured_square_x_mm=24.885, measured_square_y_mm=24.885),
        measured=True,
    ).getChessboardCorners()

    ratio = 24.885 / 25.0
    assert measured[:, 0].max() == pytest.approx(nominal[:, 0].max() * ratio, rel=1e-9)
    assert measured[:, 1].max() == pytest.approx(nominal[:, 1].max() * ratio, rel=1e-9)
