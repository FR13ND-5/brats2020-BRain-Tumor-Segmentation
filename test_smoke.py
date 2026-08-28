"""Smoke test: python3 test_smoke.py — fails loudly if anything is wired wrong."""
import torch

from src.data import DEFAULT_DATA_DIR, BraTSSliceDataset, split_by_patient
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetrics
from src.models import MODELS, get_model


def test_models():
    x = torch.randn(2, 4, 240, 240)
    target = (torch.rand(2, 3, 240, 240) > 0.9).float()
    criterion = DiceBCELoss()
    for name in MODELS:
        model = get_model(name)
        logits = model(x)
        assert logits.shape == (2, 3, 240, 240), f"{name}: bad shape {logits.shape}"
        loss = criterion(logits, target)
        loss.backward()  # forward AND backward must work
        assert torch.isfinite(loss), f"{name}: non-finite loss"
        print(f"ok {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")


def test_split_and_data():
    train_files, val_files = split_by_patient(DEFAULT_DATA_DIR)
    train_vols = {f.name.split("_slice")[0] for f in train_files}
    val_vols = {f.name.split("_slice")[0] for f in val_files}
    assert not train_vols & val_vols, "patient leakage between train and val!"
    image, mask = BraTSSliceDataset(val_files[:1])[0]
    assert image.shape == (4, 240, 240) and image.dtype == torch.float32
    assert mask.shape == (3, 240, 240) and set(mask.unique().tolist()) <= {0.0, 1.0}
    image_aug, mask_aug = BraTSSliceDataset(val_files[:1], augment=True)[0]
    assert image_aug.shape == (4, 240, 240)
    print(f"ok data: {len(train_vols)} train / {len(val_vols)} val volumes, no overlap")


def test_metrics():
    m = SegmentationMetrics()
    perfect = torch.full((1, 3, 8, 8), 10.0)  # logits -> all positive
    target = torch.ones(1, 3, 8, 8)
    m.update(perfect, target)
    assert abs(m.compute()["dice"] - 1.0) < 1e-4
    print("ok metrics")


if __name__ == "__main__":
    test_models()
    test_split_and_data()
    test_metrics()
    print("all smoke tests passed")
