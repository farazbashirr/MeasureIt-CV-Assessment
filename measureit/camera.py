"""Camera intrinsics: the single source of truth for every later stage.

The file written by ``calibration.calibrate`` is loaded by the measurement
command. Nothing else in the project is allowed to hard-code a camera matrix
or a distortion vector.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

SCHEMA = "measureit.camera/1"


class CameraModelError(RuntimeError):
    """A camera file is missing, malformed, or used with the wrong image size."""


def _as_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise CameraModelError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise CameraModelError(f"{name} contains non-finite values")
    return arr


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera with OpenCV radial-tangential distortion."""

    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: np.ndarray
    rms_reprojection_px: float
    n_views_used: int
    created_utc: str
    board: dict[str, Any]
    source_images: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        width, height = (int(v) for v in self.image_size)
        if width <= 0 or height <= 0:
            raise CameraModelError(f"image_size must be positive, got {self.image_size}")
        _as_matrix(self.camera_matrix, (3, 3), "camera_matrix")
        _as_matrix(self.new_camera_matrix, (3, 3), "new_camera_matrix")

        dist = np.asarray(self.dist_coeffs, dtype=np.float64).ravel()
        if dist.size not in (4, 5, 8, 12, 14):
            raise CameraModelError(
                f"dist_coeffs must have 4, 5, 8, 12 or 14 values, got {dist.size}"
            )
        if not np.isfinite(dist).all():
            raise CameraModelError("dist_coeffs contains non-finite values")

        rms = float(self.rms_reprojection_px)
        if not np.isfinite(rms) or rms < 0:
            raise CameraModelError(f"rms_reprojection_px must be finite and >= 0, got {rms}")

        object.__setattr__(self, "image_size", (width, height))
        object.__setattr__(self, "dist_coeffs", dist)

    # -- geometry -----------------------------------------------------------

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort a BGR or grayscale image captured at ``image_size``."""
        if image.ndim not in (2, 3):
            raise CameraModelError(f"expected a 2D or 3D image, got shape {image.shape}")
        height, width = image.shape[:2]
        if (width, height) != tuple(self.image_size):
            raise CameraModelError(
                f"image is {width}x{height} but the camera was calibrated at "
                f"{self.image_size[0]}x{self.image_size[1]}; resize with the same "
                "method used for the calibration images"
            )
        return cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix
        )

    def undistort_points(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Map pixel coordinates from the distorted to the undistorted frame."""
        src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(
            src, self.camera_matrix, self.dist_coeffs, None, None, self.new_camera_matrix
        )
        return out.reshape(-1, 2)

    def max_undistort_displacement_px(self) -> float:
        """Largest pixel shift caused by undistortion, measured at the border.

        This is the honest size of the effect that item 6 has to demonstrate:
        if it is a fraction of a pixel, undistortion cannot change a
        measurement by millimetres and the report must say so.
        """
        width, height = self.image_size
        probes = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
                [width / 2, 0],
                [width / 2, height - 1],
                [0, height / 2],
                [width - 1, height / 2],
            ],
            dtype=np.float64,
        )
        mapped = cv2.undistortPoints(
            probes.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            None,
            None,
            self.camera_matrix,
        ).reshape(-1, 2)
        return float(np.linalg.norm(mapped - probes, axis=1).max())

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "created_utc": self.created_utc,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_model": f"opencv-radtan-{self.dist_coeffs.size}",
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "new_camera_matrix": self.new_camera_matrix.tolist(),
            "rms_reprojection_px": float(self.rms_reprojection_px),
            "n_views_used": int(self.n_views_used),
            "board": dict(self.board),
            "source_images": list(self.source_images),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraModel":
        if not isinstance(payload, dict):
            raise CameraModelError("camera file must contain a JSON object")
        if payload.get("schema") != SCHEMA:
            raise CameraModelError(
                f"unsupported camera schema {payload.get('schema')!r}; expected {SCHEMA!r}"
            )
        try:
            size = payload["image_size"]
            board = payload["board"]
            created = str(payload["created_utc"])
            views = int(payload["n_views_used"])
            rms = float(payload["rms_reprojection_px"])
        except KeyError as exc:
            raise CameraModelError(f"camera file is missing key {exc.args[0]!r}") from exc

        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise CameraModelError("image_size must be [width, height]")
        if not isinstance(board, dict):
            raise CameraModelError("board must be a JSON object")

        return cls(
            image_size=(int(size[0]), int(size[1])),
            camera_matrix=_as_matrix(payload["camera_matrix"], (3, 3), "camera_matrix"),
            dist_coeffs=np.asarray(payload["distortion_coefficients"], dtype=np.float64),
            new_camera_matrix=_as_matrix(
                payload["new_camera_matrix"], (3, 3), "new_camera_matrix"
            ),
            rms_reprojection_px=rms,
            n_views_used=views,
            created_utc=created,
            board=board,
            source_images=tuple(str(p) for p in payload.get("source_images", [])),
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2) + "\n"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CameraModel":
        source = Path(path)
        if not source.is_file():
            raise CameraModelError(
                f"camera file not found: {source}. Run the calibration stage first."
            )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CameraModelError(f"{source} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)
