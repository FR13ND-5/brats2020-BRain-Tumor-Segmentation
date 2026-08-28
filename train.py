"""Train one model: python3 train.py --model unet"""
import argparse
import random

import numpy as np
import torch

from src.data import DEFAULT_DATA_DIR, create_dataloaders
from src.engine import count_params, fit
from src.losses import DiceBCELoss
from src.models import MODELS, get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap files per split for a quick sanity run")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = create_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed, limit=args.limit)

    model = get_model(args.model).to(device)
    print(f"Model: {args.model} ({count_params(model) / 1e6:.1f}M params), "
          f"device: {device}")

    fit(model, args.model, train_loader, val_loader, DiceBCELoss(),
        epochs=args.epochs, lr=args.lr, device=device,
        output_dir=args.output_dir, patience=args.patience,
        config=vars(args))


if __name__ == "__main__":
    main()
