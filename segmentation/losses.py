"""Segmentation losses: binary cross-entropy combined with soft Dice."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    """0.5 * BCEWithLogits + 0.5 * (1 - soft Dice).

    BCE gives a stable pixel gradient; Dice pushes the predicted region towards
    the object even when the foreground is small. Both terms are computed on the
    same logits, and the Dice term is averaged per image so a frame with a small
    object contributes as much as a frame with a large one.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be >= 0")
        if bce_weight + dice_weight <= 0:
            raise ValueError("at least one loss weight must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} and target {tuple(target.shape)} differ"
            )
        loss = logits.new_zeros(())
        if self.bce_weight:
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, target
            )
        if self.dice_weight:
            probability = torch.sigmoid(logits)
            dims = tuple(range(1, probability.ndim))
            intersection = (probability * target).sum(dim=dims)
            union = probability.sum(dim=dims) + target.sum(dim=dims)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            loss = loss + self.dice_weight * (1 - dice).mean()
        return loss
