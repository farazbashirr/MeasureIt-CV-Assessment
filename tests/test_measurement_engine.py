"""Synthetic measurement-engine tests.

Markers and masks are generated programmatically from a known synthetic scale.
They verify the software's error handling and geometry, not real-world
accuracy against a measuring instrument.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from measurement.engine import measure
from measurement.reference import ReferenceError, ReferenceSpec

ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _scene(
    *,
    marker_px: int = 400,
    marker_at: tuple[int, int] = (100, 100),
    object_rect: tuple[int, int, int, int] | None = (700, 300, 800, 480),
    extra_object_rect: tuple[int, int, int, int] | None = None,
):
    image = np.full((1200, 1600, 3), 245, dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(DICTIONARY, 0, marker_px)
    x, y = marker_at
    image[y : y + marker_px, x : x + marker_px] = cv2.cvtColor(
        marker_image, cv2.COLOR_GRAY2BGR
    )
    mask = np.zeros((1200, 1600), dtype=np.uint8)
    for rect in (object_rect, extra_object_rect):
        if rect is not None:
            left, top, width, height = rect
            mask[top : top + height, left : left + width] = 255
    return image, mask


def test_synthetic_measurement_is_100_by_60_mm():
    image, mask = _scene()
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))

    assert result.ok
    assert result.width_mm == pytest.approx(100.0, abs=0.5)
    assert result.height_mm == pytest.approx(60.0, abs=0.5)
    assert result.reliability == "high"
    assert result.checks["reference_present"]
    assert result.checks["single_component"]
    assert result.checks["inside_frame"]
    assert result.warnings[0].startswith("marker and object are assumed coplanar")


def test_missing_reference_is_reported():
    image = np.full((1200, 1600, 3), 245, dtype=np.uint8)
    _, mask = _scene()
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert not result.ok
    assert result.code == "missing_reference"
    assert result.width_mm is None


def test_missing_object_is_reported():
    image, _ = _scene()
    empty = np.zeros((1200, 1600), dtype=np.uint8)
    result = measure(image, empty, ReferenceSpec(marker_size_mm=50.0))
    assert not result.ok
    assert result.code == "missing_object"
    assert result.marker is not None


def test_object_touching_the_border_is_partial_occlusion():
    image, mask = _scene(object_rect=(0, 300, 800, 480))
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert not result.ok
    assert result.code == "partial_occlusion"
    assert result.checks["inside_frame"] is False


def test_split_object_is_partial_occlusion():
    image, mask = _scene(
        object_rect=(600, 300, 400, 480),
        extra_object_rect=(1050, 300, 400, 480),
    )
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert not result.ok
    assert result.code == "partial_occlusion"
    assert result.components == 2


def test_unconfigured_reference_size_is_explicit():
    image, mask = _scene()
    result = measure(image, mask, ReferenceSpec(marker_size_mm=None))
    assert not result.ok
    assert result.code == "reference_size_unknown"
    assert result.checks["reference_size_configured"] is False


def test_implausible_dimensions_are_rejected():
    image, mask = _scene()
    result = measure(
        image, mask, ReferenceSpec(marker_size_mm=50.0, max_dimension_mm=10.0)
    )
    assert not result.ok
    assert result.code == "invalid_geometry"
    assert result.checks["dimensions_plausible"] is False


def test_mask_shape_mismatch_is_rejected():
    image, _ = _scene()
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert not result.ok
    assert result.code == "invalid_geometry"
    assert result.checks["mask_shape_ok"] is False


def test_medium_reliability_with_smaller_marker():
    image, mask = _scene(marker_px=100, object_rect=(700, 300, 200, 120))
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert result.ok
    assert result.reliability == "medium"
    assert any("medium reliability" in warning for warning in result.warnings)


def test_low_reliability_near_the_configured_limits():
    image, mask = _scene(marker_px=70, object_rect=(700, 300, 140, 84))
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    assert result.ok
    assert result.reliability == "low"
    assert any("low reliability" in warning for warning in result.warnings)


def test_small_object_warning():
    image, mask = _scene(object_rect=(700, 300, 40, 24))
    result = measure(
        image,
        mask,
        ReferenceSpec(marker_size_mm=50.0, min_object_area_pct=0.0),
    )
    assert result.ok
    assert any("smaller than 10 mm" in warning for warning in result.warnings)


def test_result_is_json_serialisable():
    image, mask = _scene()
    result = measure(image, mask, ReferenceSpec(marker_size_mm=50.0))
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert '"width_mm"' in payload
    assert '"coplanarity_assumption"' in payload


def test_committed_reference_template_is_not_usable():
    spec = ReferenceSpec.load(ROOT / "measurement" / "reference.json")
    assert spec.marker_size_mm is None
    assert spec.usable is False
    with pytest.raises(ReferenceError):
        spec.require_usable()


def test_reference_rejects_unknown_keys():
    with pytest.raises(ReferenceError):
        ReferenceSpec.from_dict({"schema": "measureit.reference/1", "marker_siz_mm": 50})
