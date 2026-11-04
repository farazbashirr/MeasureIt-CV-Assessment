"""ChArUco detection and calibration, independent of the command line.

The functions here are deliberately pure so the maths can be exercised with
synthetic correspondences that have a known answer. Nothing in this module
writes a camera file; that is the caller's decision after the gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from measureit.io import ImageLoadError, load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CalibrationError(RuntimeError):
    """Base class for calibration failures with a user-facing reason."""


class NoImagesError(CalibrationError):
    pass


class NoUsableViewsError(CalibrationError):
    pass


@dataclass
class View:
    """One candidate image and everything we learned about it."""

    path: Path
    image_size: tuple[int, int] | None = None
    n_corners: int = 0
    n_markers: int = 0
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    used: bool = False
    reason: str = ""
    rms_px: float | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def detected(self) -> bool:
        return self.corners is not None


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_reprojection_px: float
    view_rms: list[float]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    used: list[View]
    rejected: list[View]
    image_size: tuple[int, int] = field(default=(0, 0))


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise NoImagesError(
            f"{root} does not exist. Copy the original calibration JPEGs there first."
        )
    images = sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if not images:
        raise NoImagesError(
            f"{root} contains no images ({', '.join(sorted(IMAGE_SUFFIXES))})"
        )
    return images


def detect_view(
    path: Path,
    detector: cv2.aruco.CharucoDetector,
    min_corners: int,
    working_size: tuple[int, int] | None = None,
) -> View:
    try:
        image = load_image(path, working_size)
    except ImageLoadError as exc:
        return View(path=path, reason=str(exc))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    view = View(path=path, image_size=(width, height))

    try:
        corners, ids, _, marker_ids = detector.detectBoard(image)
    except cv2.error as exc:  # pragma: no cover - defensive, OpenCV raises rarely
        view.reason = f"OpenCV detection error: {exc}"
        return view

    view.n_markers = 0 if marker_ids is None else int(len(marker_ids))
    if corners is None or ids is None or len(ids) == 0:
        view.reason = "no ChArUco corners detected"
        return view

    view.n_corners = int(len(ids))
    if view.n_corners < min_corners:
        view.reason = f"only {view.n_corners} inner corners (< {min_corners})"
        return view

    view.corners = corners.reshape(-1, 2).astype(np.float64)
    view.ids = ids.reshape(-1).astype(np.int32)
    return view


def calibrate_correspondences(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Thin, testable wrapper around ``cv2.calibrateCamera``."""
    if len(object_points) != len(image_points):
        raise CalibrationError("object and image point lists must have equal length")
    if len(object_points) < 3:
        raise CalibrationError("at least three views are needed to calibrate")

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in object_points],
        [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points],
        image_size,
        None,
        None,
        flags=0,
    )
    return (
        float(rms),
        np.asarray(matrix, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).ravel(),
        [np.asarray(r, dtype=np.float64).ravel() for r in rvecs],
        [np.asarray(t, dtype=np.float64).ravel() for t in tvecs],
    )


def reprojection_errors(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            np.asarray(obj, dtype=np.float64),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        delta = projected.reshape(-1, 2) - np.asarray(img, dtype=np.float64).reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def calibrate_views(
    views: Iterable[View],
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
    *,
    min_views: int,
    max_view_rms: float,
    max_reject_fraction: float = 0.25,
    max_rounds: int = 8,
) -> CalibrationResult:
    """Calibrate on detected views, dropping the worst view one at a time.

    Dropping one view per round (rather than everything above a threshold at
    once) keeps the rejection explainable: every rejected file gets a reason
    that quotes the error that removed it.
    """
    candidates = [
        v for v in views if v.detected and v.image_size == tuple(image_size)
    ]
    if len(candidates) < min_views:
        raise NoUsableViewsError(
            f"only {len(candidates)} usable views for a {image_size[0]}x{image_size[1]} "
            f"calibration; at least {min_views} are required"
        )

    object_catalog = board.getChessboardCorners()
    rejected: list[View] = []
    budget = int(max_reject_fraction * len(candidates))

    result: CalibrationResult | None = None
    for _ in range(max_rounds + 1):
        object_points = [object_catalog[v.ids] for v in candidates]
        image_points = [v.corners for v in candidates]

        rms, matrix, dist, rvecs, tvecs = calibrate_correspondences(
            object_points, image_points, image_size
        )
        per_view = reprojection_errors(
            object_points, image_points, rvecs, tvecs, matrix, dist
        )

        for view, error in zip(candidates, per_view):
            view.rms_px = error

        result = CalibrationResult(
            camera_matrix=matrix,
            dist_coeffs=dist,
            rms_reprojection_px=rms,
            view_rms=per_view,
            rvecs=rvecs,
            tvecs=tvecs,
            used=list(candidates),
            rejected=list(rejected),
            image_size=tuple(image_size),
        )

        median = float(np.median(per_view))
        threshold = max(max_view_rms, 3.0 * median)
        worst = int(np.argmax(per_view))

        if per_view[worst] <= threshold or len(candidates) <= min_views or len(rejected) >= budget:
            break

        dropped = candidates.pop(worst)
        dropped.used = False
        dropped.reason = (
            f"reprojection rms {per_view[worst]:.2f} px exceeds threshold "
            f"{threshold:.2f} px (median {median:.2f} px)"
        )
        rejected.append(dropped)

    assert result is not None
    for view in candidates:
        view.used = True
        view.reason = ""
    for view in rejected:
        view.used = False
    return result
