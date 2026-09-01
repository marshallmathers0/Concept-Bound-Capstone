"""
Training loop for the Pedagogical Boundary Detector (Stage 1).

Usage
-----
    python -m src.stage1_boundary.train \
        --features-dir data/features \
        --val-split 0.2 \
        --checkpoint checkpoints/stage1_boundary_detector.pt

The script:
  1. Loads all `<video_id>.npz` feature files from `--features-dir`.
  2. Splits videos into train/val sets.
  3. Trains the `PedagogicalBoundaryDetector` with the contrastive boundary
     loss (margin loss on adjacent pairs + segment InfoNCE).
  4. Saves the best checkpoint (lowest validation loss) to `--checkpoint`.
"""

import argparse
import glob
import os
import random

import torch
from torch.utils.data import DataLoader

from config import FEATURES, STAGE1, PATHS, ensure_dirs
from src.stage1_boundary.dataset import BoundaryDataset, collate_fn
from src.stage1_boundary.model import PedagogicalBoundaryDetector
from src.stage1_boundary.losses import contrastive_boundary_loss


def list_video_ids(features_dir: str):
    paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def split_video_ids(video_ids, val_split=0.2, seed=42):
    rng = random.Random(seed)
    ids = list(video_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_split)) if len(ids) > 1 else 0
    return ids[n_val:], ids[:n_val]


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        features = batch["features"].to(device)
        is_boundary = batch["is_boundary"].to(device)
        seg_ids = batch["seg_ids"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)

        with torch.set_grad_enabled(train):
            z = model(features, key_padding_mask=key_padding_mask)
            loss = contrastive_boundary_loss(
                z, is_boundary, seg_ids,
                margin=STAGE1.margin,
                num_negatives=STAGE1.window_neg_samples,
                key_padding_mask=key_padding_mask,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=PATHS.features_dir)
    parser.add_argument("--checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=STAGE1.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE1.batch_size)
    parser.add_argument("--lr", type=float, default=STAGE1.lr)
    parser.add_argument("--encoder-type", default=STAGE1.encoder_type, choices=["cnn", "transformer", "bilstm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dirs()

    video_ids = list_video_ids(args.features_dir)
    if len(video_ids) == 0:
        raise RuntimeError(f"No feature files found in {args.features_dir}. "
                            f"Run feature extraction first.")

    train_ids, val_ids = split_video_ids(video_ids, args.val_split)
    print(f"Train videos: {len(train_ids)} | Val videos: {len(val_ids)}")

    train_ds = BoundaryDataset(train_ids, args.features_dir)
    val_ds = BoundaryDataset(val_ids, args.features_dir) if val_ids else None

    pin_memory = args.device.startswith('cuda') if isinstance(args.device, str) else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory)
                   if val_ds else None)

    # Infer the per-time-step feature dimensionality from the actual data,
    # rather than assuming the default CLIP dimension in config.py. This keeps
    # the script robust to different real feature extraction setups.
    input_dim = train_ds[0]["features"].shape[1]
    print(f"Detected per-time-step feature dimension: {input_dim}")

    model = PedagogicalBoundaryDetector(
        input_dim=input_dim,
        hidden_dim=STAGE1.hidden_dim,
        embed_dim=STAGE1.embed_dim,
        encoder_type=args.encoder_type,
        num_layers=STAGE1.num_layers,
        num_heads=STAGE1.num_heads,
        dropout=STAGE1.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, train=True)
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, args.device, train=False)
        else:
            val_loss = train_loss

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": STAGE1.hidden_dim,
                    "embed_dim": STAGE1.embed_dim,
                    "encoder_type": args.encoder_type,
                    "num_layers": STAGE1.num_layers,
                    "num_heads": STAGE1.num_heads,
                    "dropout": STAGE1.dropout,
                },
            }, args.checkpoint)
            print(f"  -> saved best checkpoint to {args.checkpoint} (val_loss={val_loss:.4f})")

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()
