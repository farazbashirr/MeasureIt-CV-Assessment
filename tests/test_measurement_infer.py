"""Synthetic inference tests.

The camera and reference here are explicit synthetic fixtures: a pinhole camera
with zero distortion and a marker size chosen for the test. These tests verify
the refusal paths, the CPU preprocessing/refinement mechanics, and the JSON /
overlay plumbing. They are not evidence of real-world measurement accuracy.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from measureit.camera import CameraModel
from measurement.infer import (
    EXIT_UNAVAILABLE,
    analyze_frame,
    draw_overlay,
    main,
    preprocess,
    run_inference,
    segment_with_model,
)
from measurement.reference import ReferenceSpec
from segmentation.model import IMAGENET_MEAN, IMAGENET_STD

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _camera() -> CameraModel:
    matrix = np.array(
        [[1200.0, 0.0, 800.0], [0.0, 1200.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraModel(
        image_size=(1600, 1200),
        camera_matrix=matrix,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        new_camera_matrix=matrix,
        rms_reprojection_px=0.0,
        n_views_used=1,
        created_utc="synthetic-test-fixture",
        board={"synthetic_test": True},
    )


def _scene():
    image = np.full((1200, 1600, 3), 245, dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(DICTIONARY, 0, 400)
    image[100:500, 100:500] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((1200, 1600), dtype=np.uint8)
    mask[300:780, 700:1500] = 255
    return image, mask


def _write_image(tmp_path: Path) -> Path:
    image, _ = _scene()
    path = tmp_path / "scene.png"
    assert cv2.imwrite(str(path), image)
    return path


def _write_reference(tmp_path: Path, size_mm: float | None) -> Path:
    path = tmp_path / "reference.json"
    payload = ReferenceSpec(marker_size_mm=size_mm).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analyze_frame_produces_known_synthetic_dimensions():
    image, mask = _scene()
    analysis = analyze_frame(
        image,
        camera=_camera(),
        reference=ReferenceSpec(marker_size_mm=50.0),
        segment_fn=lambda _: mask,
    )
    assert analysis.result.ok
    assert analysis.result.width_mm == pytest.approx(100.0, abs=0.5)
    assert analysis.result.height_mm == pytest.approx(60.0, abs=0.5)
    assert analysis.timings["analysis_total"] >= analysis.timings["undistort"]
    assert analysis.undistorted.shape == image.shape


def test_analyze_frame_reports_missing_reference():
    _, mask = _scene()
    blank = np.full((1200, 1600, 3), 245, dtype=np.uint8)
    analysis = analyze_frame(
        blank,
        camera=_camera(),
        reference=ReferenceSpec(marker_size_mm=50.0),
        segment_fn=lambda _: mask,
    )
    assert not analysis.result.ok
    assert analysis.result.code == "missing_reference"


def test_preprocess_shape():
    image, _ = _scene()
    tensor = preprocess(image, 64, IMAGENET_MEAN, IMAGENET_STD)
    assert tensor.shape == (1, 3, 64, 64)
    assert torch.isfinite(tensor).all()


def test_segment_with_model_returns_full_resolution_binary_mask():
    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])

    image, _ = _scene()
    mask = segment_with_model(
        ZeroModel(),
        image,
        image_size=64,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        refine=True,
    )
    assert mask.shape == image.shape[:2]
    assert set(int(value) for value in np.unique(mask)).issubset({0, 255})


def test_run_inference_refuses_without_camera(tmp_path):
    image_path = _write_image(tmp_path)
    output = run_inference(
        image_path=image_path,
        camera_path=tmp_path / "missing_camera.json",
        reference_path=tmp_path / "missing_reference.json",
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["status"] == "unavailable"
    assert output.payload["code"] == "missing_calibration"
    assert output.payload["measurement"] is None
    assert output.result_path is not None
    written = json.loads(output.result_path.read_text(encoding="utf-8"))
    assert written["code"] == "missing_calibration"


def test_run_inference_refuses_when_reference_size_is_null(tmp_path):
    image_path = _write_image(tmp_path)
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = _write_reference(tmp_path, None)

    output = run_inference(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "reference_size_unknown"
    assert output.payload["camera"]["n_views_used"] == 1
    assert output.payload["reference"]["usable"] is False


def test_run_inference_refuses_without_checkpoint(tmp_path):
    image_path = _write_image(tmp_path)
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = _write_reference(tmp_path, 50.0)

    output = run_inference(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "missing_checkpoint"
    assert output.payload["reference"]["usable"] is True


def test_run_inference_rejects_a_non_checkpoint_file(tmp_path):
    image_path = _write_image(tmp_path)
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = _write_reference(tmp_path, 50.0)
    checkpoint = tmp_path / "not_a_checkpoint.pt"
    checkpoint.write_text("not a torch checkpoint", encoding="utf-8")

    output = run_inference(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "invalid_checkpoint"


def test_cli_returns_unavailable_exit_code(tmp_path):
    image_path = _write_image(tmp_path)
    code = main(
        [
            "--image",
            str(image_path),
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
    assert code == EXIT_UNAVAILABLE


def test_draw_overlay_changes_the_image():
    image, mask = _scene()
    analysis = analyze_frame(
        image,
        camera=_camera(),
        reference=ReferenceSpec(marker_size_mm=50.0),
        segment_fn=lambda _: mask,
    )
    overlay = draw_overlay(
        analysis.undistorted, analysis.result, ReferenceSpec(marker_size_mm=50.0)
    )
    assert overlay.shape == analysis.undistorted.shape
    assert not np.array_equal(overlay, analysis.undistorted)
