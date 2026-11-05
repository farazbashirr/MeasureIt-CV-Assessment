"""Synthetic geometry tests.

These are software tests with generated markers and known synthetic
transformations. They verify the geometry code, not real-world measurement
accuracy against a physical instrument.
"""

import cv2
import numpy as np
import pytest

from measurement.geometry import (
    GeometryError,
    oriented_box,
    points_to_mm,
    points_to_pixels,
    reference_homography,
)
from measurement.reference import (
    Marker,
    ReferenceError,
    ReferenceSpec,
    detect_marker,
    validate_marker,
)


def _marker(corners) -> Marker:
    corners = np.asarray(corners, dtype=np.float64)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    area = float(cv2.contourArea(corners.astype(np.float32)))
    return Marker(
        marker_id=0,
        dictionary="DICT_4X4_50",
        corners_px=corners,
        side_lengths_px=tuple(float(value) for value in sides),
        side_ratio=float(sides.max() / sides.min()),
        area_px=area,
        center_px=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
    )


def _scene_with_marker(size_px: int = 400, at=(600, 400)):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 0, size_px)
    scene = np.full((1200, 1600), 255, dtype=np.uint8)
    x, y = at
    scene[y : y + size_px, x : x + size_px] = marker_image
    return scene


def test_aruco_marker_is_detected_with_expected_corners():
    scene = _scene_with_marker()
    marker = detect_marker(scene, ReferenceSpec(marker_size_mm=50.0))
    assert marker is not None
    assert marker.marker_id == 0
    np.testing.assert_allclose(
        marker.corners_px,
        [[600, 400], [1000, 400], [1000, 800], [600, 800]],
        atol=1.0,
    )
    assert all(abs(side - 400.0) < 1.0 for side in marker.side_lengths_px)


def test_missing_marker_returns_none():
    blank = np.full((1200, 1600), 255, dtype=np.uint8)
    assert detect_marker(blank, ReferenceSpec(marker_size_mm=50.0)) is None


def test_duplicate_marker_ids_are_rejected():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 0, 300)
    scene = np.full((1200, 1600), 255, dtype=np.uint8)
    scene[100:400, 100:400] = marker_image
    scene[100:400, 800:1100] = marker_image
    with pytest.raises(ReferenceError, match="appears 2 times"):
        detect_marker(scene, ReferenceSpec(marker_size_mm=50.0))


def test_clipped_marker_is_rejected():
    corners = np.array([[1, 1], [399, 1], [399, 399], [1, 399]], dtype=np.float64)
    with pytest.raises(ReferenceError, match="border"):
        validate_marker(corners, ReferenceSpec(marker_size_mm=50.0), (1200, 1600))


def test_too_small_marker_is_rejected():
    corners = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.float64)
    with pytest.raises(ReferenceError, match="too small"):
        validate_marker(corners, ReferenceSpec(marker_size_mm=50.0), (1200, 1600))


def test_foreshortened_marker_is_rejected():
    corners = np.array(
        [[100, 100], [500, 100], [480, 180], [120, 180]], dtype=np.float64
    )
    with pytest.raises(ReferenceError, match="foreshortened"):
        validate_marker(corners, ReferenceSpec(marker_size_mm=50.0), (1200, 1600))


def test_homography_maps_marker_to_metric_square():
    marker = _marker([[600, 400], [1000, 400], [1000, 800], [600, 800]])
    homography = reference_homography(marker, 50.0)
    mapped = points_to_mm(np.array([[800, 600]]), homography)
    np.testing.assert_allclose(mapped, [[25.0, 25.0]], atol=1e-6)
    back = points_to_pixels(mapped, homography)
    np.testing.assert_allclose(back, [[800.0, 600.0]], atol=1e-6)


def test_moderate_tilt_still_recovers_true_object_size():
    world_marker = np.array(
        [[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32
    )
    image_marker = np.array(
        [[200, 150], [800, 180], [780, 700], [220, 720]], dtype=np.float32
    )
    world_to_image = cv2.getPerspectiveTransform(world_marker, image_marker)

    world_object = np.array(
        [[100, 20], [200, 20], [200, 80], [100, 80]], dtype=np.float32
    )
    image_object = cv2.perspectiveTransform(
        world_object.reshape(-1, 1, 2), world_to_image
    ).reshape(-1, 2)

    homography = reference_homography(_marker(image_marker), 50.0)
    object_mm = points_to_mm(image_object, homography)
    box = oriented_box(object_mm)

    assert box.width_mm == pytest.approx(100.0, abs=1e-3)
    assert box.height_mm == pytest.approx(60.0, abs=1e-3)


def test_rotated_rectangle_dimensions_and_angle():
    width, height, angle = 100.0, 60.0, 30.0
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    radians = np.radians(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    box = oriented_box(corners @ rotation.T)

    assert box.width_mm == pytest.approx(width, abs=1e-3)
    assert box.height_mm == pytest.approx(height, abs=1e-3)
    assert box.angle_deg == pytest.approx(angle, abs=1e-3)


def test_width_and_height_are_sorted_long_side_first():
    rectangle = np.array([[0, 0], [40, 0], [40, 100], [0, 100]], dtype=np.float64)
    box = oriented_box(rectangle)
    assert box.width_mm == pytest.approx(100.0, abs=1e-6)
    assert box.height_mm == pytest.approx(40.0, abs=1e-6)


def test_degenerate_contour_is_rejected():
    collinear = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float64)
    with pytest.raises(GeometryError, match="degenerate"):
        oriented_box(collinear)


def test_degenerate_marker_homography_is_rejected():
    collinear = _marker([[0, 0], [10, 0], [20, 0], [30, 0]])
    with pytest.raises(GeometryError):
        reference_homography(collinear, 50.0)


def test_zero_reference_size_is_rejected():
    marker = _marker([[600, 400], [1000, 400], [1000, 800], [600, 800]])
    with pytest.raises(GeometryError):
        reference_homography(marker, 0.0)
