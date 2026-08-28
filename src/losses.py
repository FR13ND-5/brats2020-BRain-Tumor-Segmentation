"""Segmentation losses."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss, computed per channel and averaged."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """0.5 * Dice + 0.5 * BCE — the standard combo for class-imbalanced segmentation."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets)
                + (1 - self.dice_weight) * self.bce(logits, targets))
