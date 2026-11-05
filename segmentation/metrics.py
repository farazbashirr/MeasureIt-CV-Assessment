"""Segmentation metrics computed from probability maps.

Definitions used here (single class, one object per image):

* ``iou`` - mean per-image IoU of the binary mask thresholded at 0.5.
* ``precision``/``recall`` - pixel-level counts aggregated over the whole split.
* ``map50``/``map50_95`` - mask average precision over images. Each image
  contributes one prediction ranked by its mean probability inside the
  predicted mask; a prediction is a true positive when its mask IoU against the
  ground truth reaches the threshold. AP is the all-point interpolated area
  under the precision-recall curve, and ``map50_95`` averages AP over
  IoU 0.50, 0.55, ..., 0.95. This is **not** COCO box AP and is reported as a
  mask AP.

If a split has no annotated images the result is explicitly ``available=False``
with a reason, never a fabricated zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class MetricResult:
    available: bool
    reason: str | None
    n_images: int
    mean_iou: float | None
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    union = np.count_nonzero(pred | truth)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & truth) / union)


def pixel_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    return (
        int(np.count_nonzero(pred & truth)),
        int(np.count_nonzero(pred & ~truth)),
        int(np.count_nonzero(~pred & truth)),
    )


def average_precision(
    records: Sequence[tuple[float, float]], iou_threshold: float
) -> float:
    """All-point interpolated AP for one class with one object per image."""
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, iou in ordered:
        if iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
        recalls.append(true_positives / len(ordered))
        denominator = true_positives + false_positives
        precisions.append(true_positives / denominator if denominator else 0.0)

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    return float(
        sum(
            (mrec[index + 1] - mrec[index]) * mpre[index + 1]
            for index in range(len(mrec) - 1)
        )
    )


def compute_metrics(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    threshold: float = 0.5,
) -> MetricResult:
    if len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have the same length")
    if not probabilities:
        return MetricResult(
            available=False,
            reason="no annotated images in this split",
            n_images=0,
            mean_iou=None,
            precision=None,
            recall=None,
            map50=None,
            map50_95=None,
        )

    ious: list[float] = []
    records: list[tuple[float, float]] = []
    total_tp = total_fp = total_fn = 0

    for probability, target in zip(probabilities, targets):
        prediction = probability >= threshold
        truth = target > 0.5
        iou = binary_iou(prediction, truth)
        ious.append(iou)

        if prediction.any():
            confidence = float(probability[prediction].mean())
        else:
            confidence = 0.0
        records.append((confidence, iou))

        tp, fp, fn = pixel_counts(prediction, truth)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    map50 = average_precision(records, 0.50)
    map50_95 = float(
        np.mean([average_precision(records, threshold) for threshold in IOU_THRESHOLDS])
    )

    return MetricResult(
        available=True,
        reason=None,
        n_images=len(ious),
        mean_iou=float(np.mean(ious)),
        precision=float(precision),
        recall=float(recall),
        map50=float(map50),
        map50_95=float(map50_95),
    )
