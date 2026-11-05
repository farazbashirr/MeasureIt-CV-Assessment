"""Synthetic tests for the raw-vs-undistorted comparison tool.

The camera, marker and mask are explicit test fixtures. They verify the
software plumbing and difference arithmetic, not real-world accuracy.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.raw_vs_undistorted import (
    compare_frame,
    main,
    measurement_difference,
    run_comparison,
    side_by_side,
)
from measureit.camera import CameraModel
from measurement.engine import MeasurementResult
from measurement.infer import EXIT_UNAVAILABLE
from measurement.reference import ReferenceSpec

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _camera(*, distorted: bool = True) -> CameraModel:
    matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = (
        np.array([0.08, -0.03, 0.0005, -0.0005, 0.0], dtype=np.float64)
        if distorted
        else np.zeros(5, dtype=np.float64)
    )
    return CameraModel(
        image_size=(640, 480),
        camera_matrix=matrix,
        dist_coeffs=distortion,
        new_camera_matrix=matrix.copy(),
        rms_reprojection_px=0.2,
        n_views_used=30,
        created_utc="synthetic-test-fixture",
        board={"synthetic_test": True},
    )


def _scene():
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(DICTIONARY, 0, 150)
    image[20:170, 20:170] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[180:300, 280:520] = 255
    return image, mask


def _write_reference(tmp_path: Path, size_mm: float | None) -> Path:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(ReferenceSpec(marker_size_mm=size_mm).to_dict()), encoding="utf-8"
    )
    return path


def _write_image(tmp_path: Path) -> Path:
    image, _ = _scene()
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_compare_frame_measures_both_versions_and_reports_difference():
    image, mask = _scene()
    comparison = compare_frame(
        image,
        camera=_camera(),
        reference=ReferenceSpec(marker_size_mm=40.0),
        segment_fn=lambda _: mask,
    )
    assert comparison.raw_result.ok
    assert comparison.undistorted_result.ok
    assert comparison.difference["available"] is True
    assert comparison.difference["definition"].startswith("undistorted minus raw")
    assert comparison.difference["width_delta_mm"] == pytest.approx(
        comparison.undistorted_result.width_mm - comparison.raw_result.width_mm
    )
    assert comparison.undistorted.shape == image.shape


def test_measurement_difference_is_unavailable_when_one_side_rejects():
    ok = MeasurementResult(
        status="ok",
        code=None,
        message="ok",
        width_mm=100.0,
        height_mm=60.0,
    )
    rejected = MeasurementResult(
        status="rejected",
        code="missing_reference",
        message="no marker",
    )
    difference = measurement_difference(ok, rejected)
    assert difference["available"] is False
    assert "must succeed" in difference["reason"]
    assert difference["undistorted"]["code"] == "missing_reference"


def test_side_by_side_has_double_width_plus_gap():
    image, _ = _scene()
    combined = side_by_side(image, image)
    assert combined.shape[0] == image.shape[0]
    assert combined.shape[1] == image.shape[1] * 2 + 12


def test_run_comparison_without_calibration_writes_unavailable_json(tmp_path):
    image_path = _write_image(tmp_path)
    output = run_comparison(
        image_path=image_path,
        camera_path=tmp_path / "missing_camera.json",
        reference_path=tmp_path / "missing_reference.json",
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "missing_calibration"
    assert output.payload["images_available"] is False
    assert output.payload["measurements"]["raw"] is None
    assert output.json_path is not None
    written = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert written["code"] == "missing_calibration"


def test_run_comparison_without_checkpoint_still_writes_raw_and_undistorted(tmp_path):
    image_path = _write_image(tmp_path)
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = _write_reference(tmp_path, 40.0)

    output = run_comparison(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "missing_checkpoint"
    assert output.payload["images_available"] is True
    assert output.payload["measurements_available"] is False
    assert output.payload["measurements"]["raw"]["code"] == "missing_checkpoint"
    assert output.payload["difference"]["available"] is False
    assert output.raw_path is not None and output.raw_path.is_file()
    assert output.undistorted_path is not None and output.undistorted_path.is_file()
    assert output.side_by_side_path is not None and output.side_by_side_path.is_file()


def test_run_comparison_reports_unknown_reference_size(tmp_path):
    image_path = _write_image(tmp_path)
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = _write_reference(tmp_path, None)

    output = run_comparison(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "reference_size_unknown"
    assert output.payload["reference"]["usable"] is False
    assert output.payload["images_available"] is True


def test_run_comparison_rejects_wrong_image_size(tmp_path):
    image_path = _write_image(tmp_path)
    camera = _camera()
    other = CameraModel(
        image_size=(1600, 1200),
        camera_matrix=camera.camera_matrix,
        dist_coeffs=camera.dist_coeffs,
        new_camera_matrix=camera.new_camera_matrix,
        rms_reprojection_px=0.2,
        n_views_used=30,
        created_utc="synthetic-test-fixture",
        board={"synthetic_test": True},
    )
    camera_path = tmp_path / "camera.json"
    other.save(camera_path)

    output = run_comparison(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=_write_reference(tmp_path, 40.0),
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "calibration_image_size_mismatch"
    assert output.payload["images_available"] is False


def test_cli_returns_usage_code_for_unreadable_image(tmp_path):
    code = main(
        [
            "--image",
            str(tmp_path / "missing.png"),
            "--camera",
            str(tmp_path / "missing_camera.json"),
            "--reference",
            str(tmp_path / "missing_reference.json"),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
