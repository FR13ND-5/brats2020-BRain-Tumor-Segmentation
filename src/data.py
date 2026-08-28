"""BraTS2020 slice dataset: loading, patient-wise split, dataloaders."""
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
# Verified from the data: the 3 mask channels are disjoint one-hot labels
# (BraTS labels 1, 2, 4), NOT the nested WT/TC/ET regions.
CLASSES = ["NCR/NET", "ED", "ET"]

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset/brats2020/BraTS2020_training_data/content/data"
)


class BraTSSliceDataset(Dataset):
    """One 240x240 axial slice per item: image (4,H,W) float32, mask (3,H,W) float32."""

    def __init__(self, files, augment=False):
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            image = f["image"][:].astype(np.float32)  # (H, W, 4), z-scored
            mask = f["mask"][:].astype(np.float32)    # (H, W, 3), binary
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        if self.augment:
            image, mask = self._augment(image, mask)
        return image, mask

    @staticmethod
    def _augment(image, mask):
        if torch.rand(1) < 0.5:
            image, mask = torch.flip(image, [2]), torch.flip(mask, [2])
        if torch.rand(1) < 0.5:
            image, mask = torch.flip(image, [1]), torch.flip(mask, [1])
        k = int(torch.randint(0, 4, (1,)))
        if k:
            image, mask = torch.rot90(image, k, [1, 2]), torch.rot90(mask, k, [1, 2])
        return image, mask


def split_by_patient(data_dir, val_fraction=0.2, seed=42):
    """Split by patient volume, never by slice, to avoid train/val leakage."""
    by_volume = {}
    for path in sorted(Path(data_dir).glob("volume_*_slice_*.h5")):
        vol = int(re.match(r"volume_(\d+)_", path.name).group(1))
        by_volume.setdefault(vol, []).append(path)
    volumes = sorted(by_volume)
    rng = np.random.default_rng(seed)
    rng.shuffle(volumes)
    n_val = int(round(len(volumes) * val_fraction))
    val_vols, train_vols = volumes[:n_val], volumes[n_val:]
    train_files = [p for v in train_vols for p in by_volume[v]]
    val_files = [p for v in val_vols for p in by_volume[v]]
    print(f"Patient-wise split: {len(train_vols)} train / {len(val_vols)} val volumes "
          f"({len(train_files)} / {len(val_files)} slices)")
    return train_files, val_files


def create_dataloaders(data_dir, batch_size=32, num_workers=8,
                       val_fraction=0.2, seed=42, limit=None):
    train_files, val_files = split_by_patient(data_dir, val_fraction, seed)
    if limit:  # quick sanity runs: python3 train.py --limit 200
        train_files, val_files = train_files[:limit], val_files[:limit]
    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(BraTSSliceDataset(train_files, augment=True),
                              batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(BraTSSliceDataset(val_files),
                            batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader
