"""Shared training engine — every model trains through this identical loop,
so the comparison between architectures is fair."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .metrics import SegmentationMetrics


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=device == "cuda"):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(device=device)
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device, enabled=device == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        metrics.update(logits.float(), masks)
    return total_loss / len(loader), metrics.compute()


def fit(model, model_name, train_loader, val_loader, criterion, epochs, lr,
        device, output_dir, patience=5, config=None):
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    history = {"model": model_name, "params": count_params(model),
               "config": config or {}, "epochs": []}
    best_dice, epochs_without_improvement = 0.0, 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 scaler, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start

        history["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "time_sec": epoch_time, **val_metrics,
        })
        (output_dir / "train_log.json").write_text(json.dumps(history, indent=2))

        print(f"[{model_name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_metrics['dice']:.4f} ({epoch_time:.0f}s)")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            # model-prefixed name so checkpoints stay distinct when collected
            # together (e.g. uploaded as GitHub release assets)
            checkpoint_name = f"{model_name}_best.pth"
            torch.save({"model": model_name, "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_metrics": val_metrics},
                       output_dir / checkpoint_name)
            print(f"  saved {checkpoint_name} (dice={best_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stop: no improvement for {patience} epochs")
                break

    plot_history(history, output_dir / "curves.png")
    print(f"[{model_name}] done, best val dice: {best_dice:.4f}")
    return history


def plot_history(history, path):
    epochs = [e["epoch"] for e in history["epochs"]]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [e["train_loss"] for e in history["epochs"]], label="train")
    ax_loss.plot(epochs, [e["val_loss"] for e in history["epochs"]], label="val")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_loss.legend()
    ax_dice.plot(epochs, [e["dice"] for e in history["epochs"]], label="val dice")
    ax_dice.plot(epochs, [e["iou"] for e in history["epochs"]], label="val IoU")
    ax_dice.set(title="Validation metrics", xlabel="epoch")
    ax_dice.legend()
    fig.suptitle(history["model"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
