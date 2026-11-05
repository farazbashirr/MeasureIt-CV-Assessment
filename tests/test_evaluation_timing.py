"""Synthetic tests for the CPU benchmark.

The full-stage test injects a tiny deterministic segmenter and a synthetic
camera/marker. It verifies timing capture and report structure; the numbers are
not a real-world performance claim.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import evaluation.timing as timing
from evaluation.timing import collect_environment, main, run_benchmark
from measureit.camera import CameraModel
from measurement.infer import EXIT_UNAVAILABLE
from measurement.reference import ReferenceSpec

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


class CenterRectModel(torch.nn.Module):
    """Deterministic stand-in that predicts a centred rectangle."""

    def forward(self, x):
        height, width = x.shape[-2:]
        logits = torch.full((x.shape[0], 1, height, width), -20.0)
        logits[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 20.0
        return logits


def _camera() -> CameraModel:
    matrix = np.array(
        [[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraModel(
        image_size=(320, 240),
        camera_matrix=matrix,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        new_camera_matrix=matrix.copy(),
        rms_reprojection_px=0.0,
        n_views_used=1,
        created_utc="synthetic-test-fixture",
        board={"synthetic_test": True},
    )


def _scene() -> np.ndarray:
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(DICTIONARY, 0, 80)
    image[10:90, 10:90] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def _write_inputs(tmp_path: Path, *, size_mm: float | None = 40.0) -> tuple[Path, Path, Path, Path]:
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), _scene())
    camera_path = tmp_path / "camera.json"
    _camera().save(camera_path)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(ReferenceSpec(marker_size_mm=size_mm).to_dict()), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint placeholder")
    return image_path, camera_path, reference_path, checkpoint_path


def test_collect_environment_has_reproducibility_fields():
    environment = collect_environment()
    assert environment["python"]
    assert environment["torch"]
    assert environment["opencv"]
    assert environment["numpy"]
    assert isinstance(environment["torch_threads"], int)


def test_benchmark_without_calibration_writes_unavailable_report(tmp_path):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), _scene())
    output = run_benchmark(
        image_path=image_path,
        camera_path=tmp_path / "missing_camera.json",
        reference_path=tmp_path / "missing_reference.json",
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "missing_calibration"
    assert output.payload["timings"] == {}
    assert output.json_path is not None
    assert output.markdown_path is not None
    written = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert written["status"] == "unavailable"
    assert written["scope"] == "software_timing_only_not_accuracy"


def test_benchmark_without_measured_reference_size_is_unavailable(tmp_path):
    image_path, camera_path, reference_path, checkpoint_path = _write_inputs(
        tmp_path, size_mm=None
    )
    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "reference_size_unknown"


def test_benchmark_without_checkpoint_is_unavailable(tmp_path):
    image_path, camera_path, reference_path, _ = _write_inputs(tmp_path)
    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=tmp_path / "missing.pt",
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "missing_checkpoint"


def test_benchmark_with_invalid_checkpoint_is_unavailable(tmp_path):
    image_path, camera_path, reference_path, checkpoint_path = _write_inputs(tmp_path)
    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "invalid_checkpoint"


def test_benchmark_rejects_wrong_image_size(tmp_path):
    image_path, _, reference_path, checkpoint_path = _write_inputs(tmp_path)
    camera = _camera()
    other = CameraModel(
        image_size=(640, 480),
        camera_matrix=camera.camera_matrix,
        dist_coeffs=camera.dist_coeffs,
        new_camera_matrix=camera.new_camera_matrix,
        rms_reprojection_px=0.0,
        n_views_used=1,
        created_utc="synthetic-test-fixture",
        board={"synthetic_test": True},
    )
    camera_path = tmp_path / "other_camera.json"
    other.save(camera_path)
    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "out",
    )
    assert output.exit_code == EXIT_UNAVAILABLE
    assert output.payload["code"] == "calibration_image_size_mismatch"


def test_benchmark_collects_every_stage_with_injected_segmenter(tmp_path, monkeypatch):
    image_path, camera_path, reference_path, checkpoint_path = _write_inputs(tmp_path)
    meta = {
        "architecture": {"name": "synthetic_center_rect", "classes": 1},
        "image_size": 64,
        "normalization": {},
        "epoch": 0,
    }
    monkeypatch.setattr(
        timing,
        "load_checkpoint",
        lambda _path: (CenterRectModel(), meta),
    )
    previous_threads = torch.get_num_threads()

    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "out",
        warmup=1,
        repetitions=3,
        threads=2,
        refine=False,
    )

    assert output.exit_code == 0
    assert output.payload["status"] == "ok"
    assert output.payload["measurement"]["ok"] is True
    assert output.payload["geometry_measurement_available"] is True
    for name in (
        "model_load",
        "undistort",
        "segmentation",
        "reference_detection",
        "geometry_measurement",
        "total_inference",
    ):
        summary = output.payload["timings"][name]
        assert summary["samples"] >= 1
        assert summary["median_s"] is not None
        assert summary["p95_s"] is not None
    assert output.payload["timings"]["total_inference"]["samples"] == 3
    assert output.payload["environment"]["torch_threads"] == 2
    assert output.csv_path is not None and output.csv_path.is_file()
    markdown = output.markdown_path.read_text(encoding="utf-8")
    assert "software_timing_only_not_accuracy" in markdown
    assert "geometry + measurement" in markdown
    # The benchmark must restore the process-wide torch thread setting.
    assert torch.get_num_threads() == previous_threads


def test_benchmark_reports_partial_when_frame_does_not_measure(tmp_path, monkeypatch):
    image_path, camera_path, reference_path, checkpoint_path = _write_inputs(tmp_path)
    meta = {"architecture": {"name": "zero", "classes": 1}, "image_size": 64, "normalization": {}}

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -20.0)

    monkeypatch.setattr(timing, "load_checkpoint", lambda _path: (ZeroModel(), meta))
    output = run_benchmark(
        image_path=image_path,
        camera_path=camera_path,
        reference_path=reference_path,
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "out",
        warmup=0,
        repetitions=2,
        refine=False,
    )
    assert output.exit_code == 0
    assert output.payload["status"] == "partial"
    assert output.payload["code"] == "measurement_not_ok"
    assert output.payload["timings"]["geometry_measurement"]["samples"] == 0
    assert output.payload["geometry_measurement_available"] is False


def test_cli_returns_usage_for_invalid_repetitions(tmp_path):
    image_path, camera_path, reference_path, checkpoint_path = _write_inputs(tmp_path)
    code = main(
        [
            "--image",
            str(image_path),
            "--camera",
            str(camera_path),
            "--reference",
            str(reference_path),
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(tmp_path / "out"),
            "--repetitions",
            "0",
        ]
    )
    assert code == 2
