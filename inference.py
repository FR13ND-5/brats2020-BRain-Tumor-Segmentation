"""Evaluate a trained model on the validation patients and save predictions.

python3 inference.py --model unet
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CLASSES, DEFAULT_DATA_DIR, BraTSSliceDataset, split_by_patient
from src.engine import count_params, evaluate
from src.losses import DiceBCELoss
from src.metrics import SegmentationMetrics
from src.models import MODELS, get_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def mask_to_rgb(mask):
    """(3,H,W) binary mask -> RGB image, one color per class."""
    return np.stack([mask[0], mask[1], mask[2]], axis=-1)


@torch.no_grad()
def save_sample_predictions(model, dataset, device, path, num_samples=6):
    # pick tumor-containing slices, spread across the val set
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    step = max(len(dataset) // (num_samples * 3), 1)
    shown = 0
    for idx in range(0, len(dataset), step):
        image, mask = dataset[idx]
        if mask.sum() < 100:
            continue
        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0] > 0.5).float().cpu().numpy()
        row = axes[shown]
        row[0].imshow(image[3], cmap="gray")  # FLAIR
        row[0].set_title("FLAIR")
        row[1].imshow(mask_to_rgb(mask.numpy()))
        row[1].set_title("ground truth")
        row[2].imshow(mask_to_rgb(pred))
        row[2].set_title("prediction")
        for ax in row:
            ax.axis("off")
        shown += 1
        if shown == num_samples:
            break
    fig.suptitle(f"R={CLASSES[0]}  G={CLASSES[1]}  B={CLASSES[2]}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def per_patient_metrics(model, val_files, device, batch_size=32):
    """Dice per patient volume, then mean/std across patients — the BraTS
    convention for reporting, complementing the pooled pixel metrics."""
    by_volume = defaultdict(list)
    for f in val_files:
        by_volume[int(re.match(r"volume_(\d+)_", f.name).group(1))].append(f)
    patients = []
    for vol, files in tqdm(sorted(by_volume.items()), desc="per-patient"):
        metrics = SegmentationMetrics(device=device)
        loader = DataLoader(BraTSSliceDataset(files), batch_size=batch_size)
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(images)
            metrics.update(logits.float(), masks)
        result = metrics.compute()
        patients.append({"volume": vol, "dice": result["dice"],
                         "dice_per_class": result["dice_per_class"]})
    dices = np.array([p["dice"] for p in patients])
    return {"dice_mean": float(dices.mean()), "dice_std": float(dices.std()),
            "num_patients": len(patients), "patients": patients}


@torch.no_grad()
def measure_speed(model, device, batch_size=32, iters=20):
    x = torch.randn(batch_size, 4, 240, 240, device=device)
    for _ in range(3):  # warmup
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - start) / (iters * batch_size) * 1000  # ms/slice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", default=None,
                        help="default: outputs/<model>/best.pth")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap val files for a quick sanity run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.model
    checkpoint_path = args.checkpoint or output_dir / "best.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = get_model(args.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded {checkpoint_path} (epoch {checkpoint['epoch']})")

    # same seed -> identical patient split as training; val patients are unseen
    _, val_files = split_by_patient(args.data_dir, seed=args.seed)
    if args.limit:
        val_files = val_files[:args.limit]
    val_dataset = BraTSSliceDataset(val_files)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    val_loss, metrics = evaluate(model, val_loader, DiceBCELoss(), device)
    metrics["val_loss"] = val_loss
    metrics["params"] = count_params(model)
    metrics["ms_per_slice"] = measure_speed(model, device)
    metrics["per_patient"] = per_patient_metrics(model, val_files, device,
                                                 args.batch_size)

    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_sample_predictions(model, val_dataset, device,
                            output_dir / "predictions.png", args.num_samples)

    summary = {**metrics, "per_patient": {
        k: v for k, v in metrics["per_patient"].items() if k != "patients"}}
    print(json.dumps(summary, indent=2))
    print(f"Saved {output_dir}/test_metrics.json and predictions.png")


if __name__ == "__main__":
    main()
