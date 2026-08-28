"""Segmentation metrics, accumulated over a whole epoch.

Accumulating TP/FP/FN/TN across the dataset (instead of averaging per-batch
scores) keeps metrics honest on the many tumor-free slices, where a per-batch
Dice is degenerate.
"""
import torch

from .data import CLASSES

EPS = 1e-6


class SegmentationMetrics:
    def __init__(self, num_classes=len(CLASSES), device="cpu"):
        self.stats = torch.zeros(4, num_classes, device=device)  # tp, fp, fn, tn

    @torch.no_grad()
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits) > 0.5).float()
        dims = (0, 2, 3)
        self.stats[0] += (preds * targets).sum(dims)
        self.stats[1] += (preds * (1 - targets)).sum(dims)
        self.stats[2] += ((1 - preds) * targets).sum(dims)
        self.stats[3] += ((1 - preds) * (1 - targets)).sum(dims)

    def compute(self):
        tp, fp, fn, tn = self.stats.cpu()
        dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
        iou = (tp + EPS) / (tp + fp + fn + EPS)
        sensitivity = (tp + EPS) / (tp + fn + EPS)
        specificity = (tn + EPS) / (tn + fp + EPS)
        return {
            "dice": dice.mean().item(),
            "iou": iou.mean().item(),
            "sensitivity": sensitivity.mean().item(),
            "specificity": specificity.mean().item(),
            "dice_per_class": {c: d.item() for c, d in zip(CLASSES, dice)},
        }
