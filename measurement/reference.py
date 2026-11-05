"""ArUco reference marker: configuration, detection and geometry validation.

The reference configuration is physical input, not a constant of the software.
``marker_size_mm`` is the measured side of the printed marker's outer black
square (the square whose corners OpenCV reports). It is ``null`` in the
committed template until somebody prints and measures a marker; the inference
command refuses to measure while it is null.

The marker and the object are assumed to lie on the same plane. A single view
cannot verify that assumption, so the measurement result states it explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

SCHEMA = "measureit.reference/1"

_ALLOWED_KEYS = {
    "schema",
    "dictionary",
    "marker_id",
    "marker_size_mm",
    "min_marker_px",
    "max_side_ratio",
    "min_object_area_pct",
    "min_dimension_mm",
    "max_dimension_mm",
    "border_margin_px",
    "notes",
}


class ReferenceError(RuntimeError):
    """The reference file is missing/malformed, or the marker geometry is invalid."""


@dataclass(frozen=True)
class ReferenceSpec:
    dictionary: str = "DICT_4X4_50"
    marker_id: int = 0
    marker_size_mm: float | None = None
    min_marker_px: float = 60.0
    max_side_ratio: float = 3.0
    min_object_area_pct: float = 0.2
    min_dimension_mm: float = 0.5
    max_dimension_mm: float = 2000.0
    border_margin_px: int = 2
    schema: str = SCHEMA
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ReferenceError(
                f"unsupported reference schema {self.schema!r}; expected {SCHEMA!r}"
            )
        if not isinstance(self.dictionary, str) or not hasattr(
            cv2.aruco, self.dictionary
        ):
            raise ReferenceError(
                f"unknown ArUco dictionary {self.dictionary!r}; use a cv2.aruco.DICT_* name"
            )
        if not self.dictionary.startswith("DICT_"):
            raise ReferenceError(
                f"dictionary {self.dictionary!r} is not an ArUco dictionary constant"
            )
        if int(self.marker_id) < 0:
            raise ReferenceError("marker_id must be >= 0")
        if self.marker_size_mm is not None:
            value = float(self.marker_size_mm)
            if not np.isfinite(value) or value <= 0:
                raise ReferenceError("marker_size_mm must be a positive finite number or null")
        if float(self.min_marker_px) <= 0:
            raise ReferenceError("min_marker_px must be positive")
        if not 1.0 <= float(self.max_side_ratio) <= 10.0:
            raise ReferenceError("max_side_ratio must be between 1 and 10")
        if not 0.0 <= float(self.min_object_area_pct) < 100.0:
            raise ReferenceError("min_object_area_pct must be in [0, 100)")
        if float(self.min_dimension_mm) <= 0:
            raise ReferenceError("min_dimension_mm must be positive")
        if float(self.max_dimension_mm) <= float(self.min_dimension_mm):
            raise ReferenceError("max_dimension_mm must exceed min_dimension_mm")
        if int(self.border_margin_px) < 0:
            raise ReferenceError("border_margin_px must be >= 0")

    @property
    def usable(self) -> bool:
        return self.marker_size_mm is not None and float(self.marker_size_mm) > 0

    def require_usable(self) -> float:
        if not self.usable:
            raise ReferenceError(
                "marker_size_mm is null: print the reference marker, measure the outer "
                "black square with a ruler/caliper, and put the value in the reference JSON"
            )
        return float(self.marker_size_mm)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceSpec":
        if not isinstance(payload, Mapping):
            raise ReferenceError("reference file must contain a JSON object")
        unknown = sorted(set(payload) - _ALLOWED_KEYS)
        if unknown:
            raise ReferenceError(f"unknown reference key(s): {', '.join(unknown)}")
        try:
            return cls(
                dictionary=str(payload.get("dictionary", "DICT_4X4_50")),
                marker_id=int(payload.get("marker_id", 0)),
                marker_size_mm=(
                    None
                    if payload.get("marker_size_mm") is None
                    else float(payload["marker_size_mm"])
                ),
                min_marker_px=float(payload.get("min_marker_px", 60.0)),
                max_side_ratio=float(payload.get("max_side_ratio", 3.0)),
                min_object_area_pct=float(payload.get("min_object_area_pct", 0.2)),
                min_dimension_mm=float(payload.get("min_dimension_mm", 0.5)),
                max_dimension_mm=float(payload.get("max_dimension_mm", 2000.0)),
                border_margin_px=int(payload.get("border_margin_px", 2)),
                schema=str(payload.get("schema", SCHEMA)),
                notes=str(payload.get("notes", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceError(f"invalid reference value: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceSpec":
        source = Path(path)
        if not source.is_file():
            raise ReferenceError(f"reference file not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReferenceError(f"{source} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Marker:
    marker_id: int
    dictionary: str
    corners_px: np.ndarray
    side_lengths_px: tuple[float, ...]
    side_ratio: float
    area_px: float
    center_px: tuple[float, float]

    @property
    def min_side_px(self) -> float:
        return min(self.side_lengths_px)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_id": int(self.marker_id),
            "dictionary": self.dictionary,
            "corners_px": [[float(x), float(y)] for x, y in self.corners_px],
            "side_lengths_px": [float(value) for value in self.side_lengths_px],
            "side_ratio": float(self.side_ratio),
            "area_px": float(self.area_px),
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
        }


def _side_lengths(corners: np.ndarray) -> np.ndarray:
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def validate_marker(
    corners: np.ndarray,
    spec: ReferenceSpec,
    image_shape: tuple[int, int],
) -> Marker:
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ReferenceError(f"marker corners must have shape (4, 2), got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ReferenceError("marker corners contain non-finite values")

    height, width = image_shape[:2]
    margin = int(spec.border_margin_px)
    if (
        corners[:, 0].min() < margin
        or corners[:, 1].min() < margin
        or corners[:, 0].max() > width - 1 - margin
        or corners[:, 1].max() > height - 1 - margin
    ):
        raise ReferenceError(
            "reference marker touches the image border and may be clipped; "
            "keep the whole marker inside the frame"
        )

    polygon = corners.astype(np.float32)
    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        raise ReferenceError("reference marker corners are not convex")

    area = float(cv2.contourArea(polygon))
    if area <= 0:
        raise ReferenceError("reference marker has zero area")

    sides = _side_lengths(corners)
    shortest = float(sides.min())
    if shortest < float(spec.min_marker_px):
        raise ReferenceError(
            f"reference marker is too small ({shortest:.1f} px < "
            f"{float(spec.min_marker_px):.1f} px); move closer or print a larger marker"
        )
    ratio = float(sides.max() / shortest)
    if ratio > float(spec.max_side_ratio):
        raise ReferenceError(
            f"reference marker is too foreshortened (side ratio {ratio:.2f} > "
            f"{float(spec.max_side_ratio):.2f}); reduce the viewing tilt"
        )

    return Marker(
        marker_id=int(spec.marker_id),
        dictionary=spec.dictionary,
        corners_px=corners,
        side_lengths_px=tuple(float(value) for value in sides),
        side_ratio=ratio,
        area_px=area,
        center_px=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
    )


def detect_marker(image_bgr: np.ndarray, spec: ReferenceSpec) -> Marker | None:
    """Return the configured reference marker, or ``None`` when it is absent.

    Raises ``ReferenceError`` when the marker is present but its geometry is
    unusable (too small, clipped, foreshortened, duplicated).
    """
    if image_bgr is None or image_bgr.ndim not in (2, 3):
        raise ReferenceError("reference detection needs a 2D or 3D image")
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
    elif image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    matches = [index for index, value in enumerate(ids.ravel()) if int(value) == spec.marker_id]
    if not matches:
        return None
    if len(matches) > 1:
        raise ReferenceError(
            f"reference marker id {spec.marker_id} appears {len(matches)} times; "
            "exactly one reference marker is required"
        )

    return validate_marker(corners[matches[0]].reshape(4, 2), spec, gray.shape)
