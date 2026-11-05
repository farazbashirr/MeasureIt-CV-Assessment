"""From a binary object mask and one reference marker to millimetres.

The engine never fabricates a value. Every failure mode named by the brief
(missing reference, missing object, partial occlusion, invalid geometry) is a
structured rejection with a stable code, and a successful result carries a
rule-based reliability grade plus the checks that produced it. The grade is a
geometric quality label, not a learned probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from measurement.geometry import (
    GeometryError,
    oriented_box,
    points_to_mm,
    points_to_pixels,
    reference_homography,
)
from measurement.reference import Marker, ReferenceError, ReferenceSpec, detect_marker

COPLANARITY = (
    "marker and object are assumed coplanar; a single view cannot verify this"
)
COMPONENT_FRACTION = 0.05


@dataclass
class MeasurementResult:
    status: str
    code: str | None
    message: str
    width_mm: float | None = None
    height_mm: float | None = None
    angle_deg: float | None = None
    reliability: str = "invalid"
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    marker: Marker | None = None
    mask_area_px: float = 0.0
    mask_area_pct: float = 0.0
    components: int = 0
    border_distance_px: float | None = None
    contour_px: np.ndarray | None = None
    box_corners_mm: np.ndarray | None = None
    box_corners_px: np.ndarray | None = None
    homography_reprojection_error_mm: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "angle_deg": self.angle_deg,
            "reliability": self.reliability,
            "checks": {name: bool(value) for name, value in self.checks.items()},
            "warnings": list(self.warnings),
            "marker": None if self.marker is None else self.marker.to_dict(),
            "mask_area_px": float(self.mask_area_px),
            "mask_area_pct": float(self.mask_area_pct),
            "components": int(self.components),
            "border_distance_px": (
                None if self.border_distance_px is None else float(self.border_distance_px)
            ),
            "box_corners_mm": (
                None
                if self.box_corners_mm is None
                else [[float(x), float(y)] for x, y in self.box_corners_mm]
            ),
            "box_corners_px": (
                None
                if self.box_corners_px is None
                else [[float(x), float(y)] for x, y in self.box_corners_px]
            ),
            "homography_reprojection_error_mm": self.homography_reprojection_error_mm,
            "coplanarity_assumption": COPLANARITY,
        }


def _warnings(extra: list[str]) -> list[str]:
    result = list(extra)
    if COPLANARITY not in result:
        result.insert(0, COPLANARITY)
    return result


def _rejected(
    code: str,
    message: str,
    *,
    checks: dict[str, bool] | None = None,
    marker: Marker | None = None,
    warnings: list[str] | None = None,
    mask_area_px: float = 0.0,
    mask_area_pct: float = 0.0,
    components: int = 0,
    border_distance_px: float | None = None,
) -> MeasurementResult:
    return MeasurementResult(
        status="rejected",
        code=code,
        message=message,
        checks=dict(checks or {}),
        warnings=_warnings(warnings or []),
        marker=marker,
        mask_area_px=mask_area_px,
        mask_area_pct=mask_area_pct,
        components=components,
        border_distance_px=border_distance_px,
    )


def measure(
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    reference: ReferenceSpec,
) -> MeasurementResult:
    """Measure the object described by ``mask`` against the reference marker.

    Expected failures are returned as ``status="rejected"`` with a code; only
    programming errors (a missing image) raise.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("measurement needs a 3-channel BGR image")
    height, width = image_bgr.shape[:2]
    checks: dict[str, bool] = {}

    if not reference.usable:
        checks["reference_size_configured"] = False
        return _rejected(
            "reference_size_unknown",
            "marker_size_mm is null in the reference configuration; a measured "
            "printed marker size is required before a millimetre value exists",
            checks=checks,
        )
    checks["reference_size_configured"] = True

    try:
        marker = detect_marker(image_bgr, reference)
    except ReferenceError as exc:
        checks["reference_present"] = True
        checks["reference_geometry_ok"] = False
        return _rejected(
            "invalid_geometry",
            f"reference marker is present but unusable: {exc}",
            checks=checks,
        )

    checks["reference_present"] = marker is not None
    checks["reference_geometry_ok"] = marker is not None
    if marker is None:
        return _rejected(
            "missing_reference",
            f"no ArUco marker with id {reference.marker_id} "
            f"({reference.dictionary}) was detected",
            checks=checks,
        )

    if mask is None:
        checks["mask_shape_ok"] = False
        checks["object_present"] = False
        return _rejected(
            "missing_object",
            "segmentation produced no mask",
            checks=checks,
            marker=marker,
        )
    mask = np.asarray(mask)
    if mask.shape[:2] != (height, width):
        checks["mask_shape_ok"] = False
        checks["object_present"] = False
        return _rejected(
            "invalid_geometry",
            f"mask is {mask.shape[1]}x{mask.shape[0]} but the image is "
            f"{width}x{height}",
            checks=checks,
            marker=marker,
        )
    checks["mask_shape_ok"] = True
    binary = (mask > 0).astype(np.uint8)
    checks["object_present"] = bool(binary.any())
    if not checks["object_present"]:
        return _rejected(
            "missing_object",
            "segmentation mask is empty",
            checks=checks,
            marker=marker,
        )

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int32)
    largest_area = int(areas.max()) if areas.size else 0
    significant = int(
        np.count_nonzero(areas >= max(1, int(COMPONENT_FRACTION * largest_area)))
    )
    checks["single_component"] = significant == 1

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _rejected(
            "missing_object",
            "segmentation mask has no contour",
            checks=checks,
            marker=marker,
        )
    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    area_pct = 100.0 * area_px / float(width * height)

    points = contour.reshape(-1, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    border_distance = float(
        min(xs.min(), ys.min(), (width - 1) - xs.max(), (height - 1) - ys.max())
    )
    checks["inside_frame"] = border_distance >= float(reference.border_margin_px)
    checks["area_ok"] = area_px > 0 and area_pct >= float(reference.min_object_area_pct)

    if not checks["single_component"] or not checks["inside_frame"]:
        return _rejected(
            "partial_occlusion",
            "object mask is split into significant components or touches the "
            "image border; the object may be occluded or truncated",
            checks=checks,
            marker=marker,
            mask_area_px=area_px,
            mask_area_pct=area_pct,
            components=significant,
            border_distance_px=border_distance,
        )
    if not checks["area_ok"]:
        return _rejected(
            "invalid_geometry",
            f"object area {area_pct:.3f}% is below the configured minimum "
            f"{float(reference.min_object_area_pct):.3f}%",
            checks=checks,
            marker=marker,
            mask_area_px=area_px,
            mask_area_pct=area_pct,
            components=significant,
            border_distance_px=border_distance,
        )

    try:
        homography = reference_homography(marker, reference.require_usable())
    except ReferenceError as exc:  # pragma: no cover - guarded above
        return _rejected("reference_size_unknown", str(exc), checks=checks, marker=marker)
    except GeometryError as exc:
        checks["homography_ok"] = False
        return _rejected(
            "invalid_geometry",
            f"cannot rectify the reference plane: {exc}",
            checks=checks,
            marker=marker,
            mask_area_px=area_px,
            mask_area_pct=area_pct,
            components=significant,
            border_distance_px=border_distance,
        )
    checks["homography_ok"] = True

    try:
        points_mm = points_to_mm(points, homography)
        box = oriented_box(points_mm)
        box_px = points_to_pixels(box.corners_mm, homography)
    except GeometryError as exc:
        return _rejected(
            "invalid_geometry",
            f"cannot measure the object: {exc}",
            checks=checks,
            marker=marker,
            mask_area_px=area_px,
            mask_area_pct=area_pct,
            components=significant,
            border_distance_px=border_distance,
        )

    checks["dimensions_plausible"] = (
        box.height_mm >= float(reference.min_dimension_mm)
        and box.width_mm <= float(reference.max_dimension_mm)
    )
    if not checks["dimensions_plausible"]:
        return _rejected(
            "invalid_geometry",
            f"measured {box.width_mm:.3f} x {box.height_mm:.3f} mm is outside the "
            f"configured limits [{float(reference.min_dimension_mm):.3f}, "
            f"{float(reference.max_dimension_mm):.3f}] mm; check marker_size_mm",
            checks=checks,
            marker=marker,
            mask_area_px=area_px,
            mask_area_pct=area_pct,
            components=significant,
            border_distance_px=border_distance,
        )

    large_marker = marker.min_side_px >= 2.0 * float(reference.min_marker_px)
    low_tilt = marker.side_ratio <= 1.5
    large_object = area_pct >= max(1.0, 2.0 * float(reference.min_object_area_pct))
    clear_border = border_distance >= max(8.0, 4.0 * float(reference.border_margin_px))
    checks["marker_large_enough"] = large_marker
    checks["low_tilt"] = low_tilt
    checks["object_large_enough"] = large_object
    checks["border_clear"] = clear_border

    warnings = _warnings([])
    if large_marker and low_tilt and large_object and clear_border:
        reliability = "high"
    elif (
        marker.min_side_px >= 1.25 * float(reference.min_marker_px)
        and marker.side_ratio <= 2.5
    ):
        reliability = "medium"
        warnings.append(
            "medium reliability: small reference marker, moderate perspective, "
            "small object, or little room to the frame border"
        )
    else:
        reliability = "low"
        warnings.append(
            "low reliability: geometry is close to the configured rejection limits; "
            "move closer, enlarge the marker, or reduce tilt"
        )

    if min(box.width_mm, box.height_mm) < 10.0:
        warnings.append(
            "object is smaller than 10 mm; its absolute error is dominated by "
            "marker corner localisation"
        )

    return MeasurementResult(
        status="ok",
        code=None,
        message="measurement produced from one reference marker",
        width_mm=float(box.width_mm),
        height_mm=float(box.height_mm),
        angle_deg=float(box.angle_deg),
        reliability=reliability,
        checks=checks,
        warnings=warnings,
        marker=marker,
        mask_area_px=area_px,
        mask_area_pct=area_pct,
        components=significant,
        border_distance_px=border_distance,
        contour_px=points.astype(np.float64),
        box_corners_mm=box.corners_mm,
        box_corners_px=box_px,
        homography_reprojection_error_mm=homography.reprojection_error_mm,
    )
