from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.board import BoardSpec, build_charuco
from calibration.pipeline import (
    NoUsableViewsError,
    View,
    calibrate_correspondences,
    calibrate_views,
)

IMAGE_SIZE = (1920, 1440)
K_TRUE = np.array([[2600.0, 0.0, 960.0], [0.0, 2600.0, 720.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([0.05, -0.02, 0.0008, -0.0005, 0.0])


BOARD_CENTRE = np.array([9 * 0.025 / 2.0, 7 * 0.025 / 2.0, 0.0])


def _poses() -> list[tuple[np.ndarray, np.ndarray]]:
    # (rx, ry, rz, dx, dy, z): the board's object points start at its top-left
    # corner, so the translation is shifted to put the board centre on the axis.
    specs = [
        (rx, ry, 0.0, 0.0, 0.0, 0.55)
        for rx in (-0.22, 0.0, 0.22)
        for ry in (-0.22, 0.0, 0.22)
    ]
    specs += [
        (0.15, -0.18, 0.05, -0.06, 0.04, 0.50),
        (-0.15, 0.18, -0.05, 0.06, -0.04, 0.62),
        (0.05, 0.05, 0.20, 0.03, 0.02, 0.45),
        (-0.05, -0.05, -0.20, -0.03, -0.02, 0.70),
        (0.10, 0.00, 0.10, 0.00, -0.05, 0.58),
    ]
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for rx, ry, rz, dx, dy, z in specs:
        rvec = np.array([rx, ry, rz], dtype=np.float64)
        tvec = np.array([dx - BOARD_CENTRE[0], dy - BOARD_CENTRE[1], z])
        poses.append((rvec, tvec))
    return poses


def _projected(board: cv2.aruco.CharucoBoard) -> list[tuple[np.ndarray, np.ndarray]]:
    object_points = board.getChessboardCorners()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for rvec, tvec in _poses():
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K_TRUE, D_TRUE)
        points = projected.reshape(-1, 2)
        if (
            points.min() < 20
            or points[:, 0].max() > IMAGE_SIZE[0] - 20
            or points[:, 1].max() > IMAGE_SIZE[1] - 20
        ):
            continue
        out.append((object_points.copy(), points))
    return out


def test_detects_all_corners_on_a_rendered_board():
    board = build_charuco(BoardSpec(), measured=False)
    texture = board.generateImage((9 * 75, 7 * 75))
    canvas = np.full((1000, 1200), 255, dtype=np.uint8)
    canvas[250 : 250 + 525, 200 : 200 + 675] = texture

    corners, ids, _, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(canvas)

    assert corners is not None and ids is not None
    assert len(ids) == 48
    assert marker_ids is not None and len(marker_ids) > 0


def test_calibration_recovers_known_intrinsics():
    board = build_charuco(BoardSpec(), measured=False)
    pairs = _projected(board)
    assert len(pairs) >= 12, f"only {len(pairs)} synthetic views fit in frame"

    object_points = [p[0] for p in pairs]
    image_points = [p[1] for p in pairs]
    rms, matrix, dist, _, _ = calibrate_correspondences(object_points, image_points, IMAGE_SIZE)

    assert rms < 0.1
    assert matrix[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.01)
    assert matrix[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.01)
    assert matrix[0, 2] == pytest.approx(K_TRUE[0, 2], abs=3.0)
    assert matrix[1, 2] == pytest.approx(K_TRUE[1, 2], abs=3.0)
    assert dist[0] == pytest.approx(D_TRUE[0], abs=0.03)


def test_outlier_view_is_rejected_with_a_reason():
    board = build_charuco(BoardSpec(), measured=False)
    pairs = _projected(board)[:10]
    assert len(pairs) == 10

    views: list[View] = []
    for index, (object_points, points) in enumerate(pairs):
        views.append(
            View(
                path=Path(f"view_{index:02d}.jpg"),
                image_size=IMAGE_SIZE,
                n_corners=len(object_points),
                corners=points.copy(),
                ids=np.arange(len(object_points)),
            )
        )

    rng = np.random.default_rng(7)
    views[3].corners = views[3].corners + rng.normal(0.0, 6.0, views[3].corners.shape)

    result = calibrate_views(
        views, board, IMAGE_SIZE, min_views=8, max_view_rms=1.5, max_reject_fraction=0.3
    )

    assert [v.path.name for v in result.rejected] == ["view_03.jpg"]
    assert "reprojection rms" in result.rejected[0].reason
    assert len(result.used) == 9
    assert result.rms_reprojection_px < 0.2


def test_too_few_views_is_an_error():
    board = build_charuco(BoardSpec(), measured=False)
    pairs = _projected(board)[:3]
    views = [
        View(
            path=Path(f"view_{i}.jpg"),
            image_size=IMAGE_SIZE,
            n_corners=len(obj),
            corners=points.copy(),
            ids=np.arange(len(obj)),
        )
        for i, (obj, points) in enumerate(pairs)
    ]
    with pytest.raises(NoUsableViewsError, match="at least 18 are required"):
        calibrate_views(views, board, IMAGE_SIZE, min_views=18, max_view_rms=1.5)
