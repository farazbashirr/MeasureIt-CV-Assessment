"""Plane rectification and millimetre geometry.

The reference marker's four corners are mapped to a square of
``marker_size_mm``. That homography is the only source of physical scale:
every object point is transformed through it, so an object lying on the same
plane as the marker is measured in millimetres regardless of camera tilt.
The coplanarity of marker and object is an assumption, not something this code
can prove from one view.

Coordinates in the rectified frame are ``(u, v)`` millimetres: ``u`` runs along
the marker's top edge and ``v`` along its right edge, as ordered by OpenCV
(top-left, top-right, bottom-right, bottom-left).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from measurement.reference import Marker

_EPSILON = 1e-9


class GeometryError(RuntimeError):
    """A homography or contour is degenerate and cannot yield a measurement."""


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    inverse: np.ndarray
    marker_size_mm: float
    reprojection_error_mm: float


@dataclass(frozen=True)
class OrientedBox:
    width_mm: float
    height_mm: float
    angle_deg: float
    center_mm: tuple[float, float]
    corners_mm: np.ndarray


def reference_homography(marker: Marker, marker_size_mm: float) -> HomographyResult:
    size = float(marker_size_mm)
    if not np.isfinite(size) or size <= 0:
        raise GeometryError("marker_size_mm must be positive and finite")

    source = np.asarray(marker.corners_px, dtype=np.float32)
    if source.shape != (4, 2):
        raise GeometryError(f"marker corners must be (4, 2), got {source.shape}")
    if not np.isfinite(source).all():
        raise GeometryError("marker corners contain non-finite values")

    target = np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], dtype=np.float32
    )
    try:
        matrix = cv2.getPerspectiveTransform(source, target)
    except cv2.error as exc:
        raise GeometryError(f"marker corners produced an invalid homography: {exc}") from exc
    if matrix is None or not np.isfinite(matrix).all():
        raise GeometryError("marker corners produced a non-finite homography")
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-12:
        raise GeometryError("marker corners produced a singular homography")

    reprojected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    error = float(np.abs(reprojected - target).max())
    if not np.isfinite(error) or error > 1e-3:
        raise GeometryError(
            f"homography does not map the marker to a square (error {error:.6g} mm)"
        )

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - determinant check above
        raise GeometryError("homography is not invertible") from exc

    return HomographyResult(
        matrix=matrix,
        inverse=inverse,
        marker_size_mm=size,
        reprojection_error_mm=error,
    )


def points_to_mm(points_px: np.ndarray, homography: HomographyResult) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    if not np.isfinite(points).all():
        raise GeometryError("object points contain non-finite values")
    mapped = cv2.perspectiveTransform(points, homography.matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        raise GeometryError("object points map to non-finite millimetres")
    return mapped.astype(np.float64)


def points_to_pixels(points_mm: np.ndarray, homography: HomographyResult) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, homography.inverse).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        raise GeometryError("millimetre points map to non-finite pixels")
    return mapped.astype(np.float64)


def oriented_box(points_mm: np.ndarray) -> OrientedBox:
    points = np.asarray(points_mm, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise GeometryError("an oriented rectangle needs at least three points")
    if not np.isfinite(points).all():
        raise GeometryError("object points contain non-finite values")

    hull = cv2.convexHull(points)
    if len(hull) < 3 or abs(float(cv2.contourArea(hull))) < 1e-6:
        raise GeometryError("object contour is degenerate (zero area)")

    rectangle = cv2.minAreaRect(points)
    (center_x, center_y), (side_a, side_b), _ = rectangle
    if not np.isfinite([center_x, center_y, side_a, side_b]).all():
        raise GeometryError("minimum-area rectangle is not finite")
    if side_a <= _EPSILON or side_b <= _EPSILON:
        raise GeometryError("minimum-area rectangle has a zero side")

    width, height = (side_a, side_b) if side_a >= side_b else (side_b, side_a)
    corners = cv2.boxPoints(rectangle).astype(np.float64)

    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    longest = int(np.argmax(lengths))
    vector = edges[longest]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)

    return OrientedBox(
        width_mm=float(width),
        height_mm=float(height),
        angle_deg=angle,
        center_mm=(float(center_x), float(center_y)),
        corners_mm=corners,
    )
