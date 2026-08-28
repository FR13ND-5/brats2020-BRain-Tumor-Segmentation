"""Compare all trained models: python3 compare.py

Reads outputs/<model>/train_log.json (+ test_metrics.json if inference was run)
and produces a markdown table and combined training curves for the report.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import CLASSES

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")


def main():
    runs = []
    for history_path in sorted(OUTPUT_DIR.glob("*/train_log.json")):
        history = json.loads(history_path.read_text())
        best = max(history["epochs"], key=lambda e: e["dice"])
        test_path = history_path.parent / "test_metrics.json"
        test = json.loads(test_path.read_text()) if test_path.exists() else {}
        runs.append({"name": history["model"], "params": history["params"],
                     "history": history, "best": best, "test": test})
    if not runs:
        print("No outputs/*/train_log.json found — train something first.")
        return

    header = (["Model", "Params (M)", "Val Dice", "Patient Dice (mean±std)",
               *[f"Dice {c}" for c in CLASSES],
               "IoU", "Sens", "Spec", "ms/slice", "Epochs", "s/epoch"])
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in runs:
        b, t = r["best"], r["test"]
        pp = t.get("per_patient")
        patient_dice = f"{pp['dice_mean']:.4f}±{pp['dice_std']:.4f}" if pp else "-"
        per_class = [f"{b['dice_per_class'][c]:.4f}" for c in CLASSES]
        avg_time = sum(e["time_sec"] for e in r["history"]["epochs"]) / len(r["history"]["epochs"])
        lines.append("| " + " | ".join([
            r["name"], f"{r['params'] / 1e6:.1f}", f"{b['dice']:.4f}",
            patient_dice, *per_class,
            f"{b['iou']:.4f}", f"{b['sensitivity']:.4f}", f"{b['specificity']:.4f}",
            f"{t['ms_per_slice']:.2f}" if t else "-",
            str(len(r["history"]["epochs"])), f"{avg_time:.0f}"]) + " |")
    table = "\n".join(lines)
    print(table)
    (OUTPUT_DIR / "comparison.md").write_text(table + "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        epochs = [e["epoch"] for e in r["history"]["epochs"]]
        ax.plot(epochs, [e["dice"] for e in r["history"]["epochs"]], label=r["name"])
    ax.set(title="Validation Dice per epoch", xlabel="epoch", ylabel="Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=120)
    print(f"\nSaved {OUTPUT_DIR}/comparison.md and comparison.png")


if __name__ == "__main__":
    main()
