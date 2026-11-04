import numpy as np
import pytest

from measureit.camera import CameraModel, CameraModelError

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])


def make_model(**overrides) -> CameraModel:
    values = {
        "image_size": (64, 48),
        "camera_matrix": K.copy(),
        "dist_coeffs": np.zeros(5),
        "new_camera_matrix": K.copy(),
        "rms_reprojection_px": 0.4,
        "n_views_used": 20,
        "created_utc": "2026-01-01T00:00:00Z",
        "board": {"squares_x": 9, "squares_y": 7},
    }
    values.update(overrides)
    return CameraModel(**values)


def test_round_trip(tmp_path):
    model = make_model(dist_coeffs=np.array([0.01, -0.002, 0.0003, -0.0004, 0.0]))
    path = tmp_path / "camera.json"
    model.save(path)

    loaded = CameraModel.load(path)
    np.testing.assert_allclose(loaded.camera_matrix, model.camera_matrix)
    np.testing.assert_allclose(loaded.dist_coeffs, model.dist_coeffs)
    np.testing.assert_allclose(loaded.new_camera_matrix, model.new_camera_matrix)
    assert loaded.image_size == (64, 48)
    assert loaded.rms_reprojection_px == pytest.approx(0.4)
    assert loaded.n_views_used == 20


def test_undistort_is_identity_without_distortion():
    model = make_model()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    np.testing.assert_allclose(model.undistort(image), image, atol=0)


def test_wrong_image_size_is_refused():
    model = make_model()
    with pytest.raises(CameraModelError, match="calibrated at 64x48"):
        model.undistort(np.zeros((24, 32, 3), dtype=np.uint8))


def test_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text('{"schema": "other/9"}', encoding="utf-8")
    with pytest.raises(CameraModelError, match="unsupported camera schema"):
        CameraModel.load(path)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(CameraModelError, match="not found"):
        CameraModel.load(tmp_path / "absent.json")


def test_undistort_points_identity_and_displacement():
    model = make_model()
    points = np.array([[0.0, 0.0], [63.0, 47.0], [32.0, 24.0]])
    np.testing.assert_allclose(model.undistort_points(points), points, atol=1e-9)
    assert model.max_undistort_displacement_px() == pytest.approx(0.0, abs=1e-9)


def test_displacement_is_positive_with_barrel_distortion():
    big_k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    model = make_model(
        image_size=(640, 480),
        camera_matrix=big_k,
        new_camera_matrix=big_k,
        dist_coeffs=np.array([0.05, -0.02, 0.0, 0.0, 0.0]),
    )
    # A zero displacement is impossible at the corner under barrel distortion.
    assert model.max_undistort_displacement_px() > 1.0
