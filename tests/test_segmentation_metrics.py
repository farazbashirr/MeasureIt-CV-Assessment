import numpy as np

from segmentation.metrics import (
    IOU_THRESHOLDS,
    average_precision,
    binary_iou,
    compute_metrics,
    pixel_counts,
)


def _block(size=10, y=1, x=1, height=4, width=4) -> np.ndarray:
    array = np.zeros((size, size), dtype=np.uint8)
    array[y : y + height, x : x + width] = 1
    return array


def test_perfect_prediction_scores_one():
    target = _block()
    result = compute_metrics([target.astype(np.float32)], [target])
    assert result.available
    assert result.n_images == 1
    assert result.mean_iou == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.map50 == 1.0
    assert result.map50_95 == 1.0


def test_disjoint_prediction_scores_zero():
    target = _block(x=1)
    other = _block(x=5, y=5)
    result = compute_metrics([other.astype(np.float32)], [target])
    assert result.mean_iou == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.map50 == 0.0
    assert result.map50_95 == 0.0


def test_partial_overlap_iou_and_pixel_counts():
    target = _block(x=1, y=1)
    prediction = _block(x=3, y=3)
    iou = binary_iou(prediction, target)
    assert iou == 4 / 28

    tp, fp, fn = pixel_counts(prediction, target)
    assert (tp, fp, fn) == (4, 12, 12)


def test_ap_with_one_hit_and_one_miss_is_half():
    # Standard all-point AP: the hit contributes precision 1.0 up to recall
    # 0.5, then the miss ends the curve at precision 0.5 on the same recall;
    # the appended (recall 1.0, precision 0) point closes the area at 0.5.
    target = _block(x=1, y=1)
    perfect = target.astype(np.float32)
    empty = np.zeros_like(perfect)
    result = compute_metrics([perfect, empty], [target, target])
    assert result.map50 == 0.5
    assert result.map50_95 == 0.5
    assert result.mean_iou == 0.5


def test_average_precision_is_zero_without_hits():
    assert average_precision([(0.9, 0.1), (0.8, 0.2)], 0.5) == 0.0


def test_empty_split_is_unavailable_not_zero():
    result = compute_metrics([], [])
    assert result.available is False
    assert result.reason is not None
    assert result.mean_iou is None
    assert result.precision is None
    assert result.map50 is None
    assert result.as_dict()["available"] is False


def test_iou_thresholds_cover_05_to_095():
    assert IOU_THRESHOLDS[0] == 0.5
    assert IOU_THRESHOLDS[-1] == 0.95
    assert len(IOU_THRESHOLDS) == 10
