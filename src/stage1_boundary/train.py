"""
Training loop for the Pedagogical Boundary Detector (Stage 1).

Usage
-----
    python -m src.stage1_boundary.train \
        --features-dir data/features \
        --val-split 0.2 \
        --checkpoint checkpoints/stage1_boundary_detector.pt

The script:
  1. Reads the video-level split from data/splits.json (test videos are never
     used here).
  2. Builds labels per `--label-source` (gold / pseudo / auto, see dataset.py).
  3. Trains the `PedagogicalBoundaryDetector` with the contrastive boundary
     loss (margin loss on adjacent pairs + segment InfoNCE).
  4. Selects the checkpoint by validation boundary F1 against *gold*
     boundaries when the val videos have them. Otherwise it falls back to
     validation loss on pseudo-labels and says so: that only measures how well
     the model imitates the heuristic, not boundary accuracy.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import EVAL, STAGE1, PATHS, ensure_dirs
from src.stage1_boundary.dataset import BoundaryDataset, collate_fn, LABEL_SOURCES
from src.stage1_boundary.model import PedagogicalBoundaryDetector
from src.stage1_boundary.losses import contrastive_boundary_loss
from src.stage1_boundary.segment import segments_from_scores
from src.stage1_boundary.pseudo_labels import boundaries_from_annotations
from src.evaluation.boundary_metrics import boundary_f1
from src.utils.gold import load_splits


def list_video_ids(features_dir: str):
    paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def gold_val_f1(model, val_ds, device) -> float:
    """Mean boundary F1 of the post-processed segmentation on gold val videos."""
    model.eval()
    f1s = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            z = model(item["features"].unsqueeze(0).to(device))
            scores = model.boundary_scores(z)[0].cpu().numpy()
            ts = item["timestamps"].numpy()
            segs = segments_from_scores(scores, ts)
            pred = np.array([s["end_idx"] - 1 for s in segs[:-1]], dtype=np.int64)
            gt = np.where(item["is_boundary"].numpy() == 1)[0]
            f1s.append(boundary_f1(pred, gt, EVAL.boundary_tolerance_steps)["f1"])
    return float(np.mean(f1s)) if f1s else float("nan")


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
    parser.add_argument("--label-source", default="auto", choices=LABEL_SOURCES)
    parser.add_argument("--drop-signals", nargs="*", default=[], choices=["ocr", "drift", "visual", "clip"],
                        help="Signal-removal ablation: zero these input channels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=STAGE1.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE1.batch_size)
    parser.add_argument("--lr", type=float, default=STAGE1.lr)
    parser.add_argument("--encoder-type", default=STAGE1.encoder_type, choices=["cnn", "transformer", "bilstm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dirs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_ids = list_video_ids(args.features_dir)
    if len(video_ids) == 0:
        raise RuntimeError(f"No feature files found in {args.features_dir}. "
                            f"Run feature extraction first.")

    split = load_splits(video_ids)
    train_ids, val_ids = split["train"], split["val"]

    train_ds = BoundaryDataset(train_ids, args.features_dir, args.label_source)
    val_ds = BoundaryDataset(val_ids, args.features_dir, args.label_source) if val_ids else None
    if len(train_ds) == 0:
        raise RuntimeError(f"No training videos with label source '{args.label_source}'.")
    print(f"Train videos: {len(train_ds)} {train_ds.sources}")
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None
    if val_ds is not None:
        print(f"Val videos:   {len(val_ds)} {val_ds.sources}")
    select_by_f1 = val_ds is not None and all(v == "gold" for v in val_ds.sources.values())
    if not select_by_f1:
        print("NOTE: val videos have no gold boundaries; selecting the checkpoint by loss "
              "on pseudo-labels. This does not measure boundary accuracy.")
    if "gold" not in train_ds.sources.values():
        print("NOTE: training on pseudo-labels only (no gold boundaries in the train split). "
              "Report the pseudo-label heuristic as a baseline next to this model.")

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
        drop_signals=args.drop_signals,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, train=True)
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, args.device, train=False)
        else:
            val_loss = train_loss
        val_f1 = gold_val_f1(model, val_ds, args.device) if select_by_f1 else None
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_gold_f1": val_f1})

        msg = f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        print(msg + (f" | val_gold_F1={val_f1:.3f}" if val_f1 is not None else ""))

        criterion = -val_f1 if select_by_f1 else val_loss
        if criterion < best_val:
            best_val = criterion
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
                    "drop_signals": list(args.drop_signals),
                },
                "train_meta": {
                    "label_source": args.label_source,
                    "train_sources": train_ds.sources,
                    "val_sources": val_ds.sources if val_ds is not None else {},
                    "selected_by": "val_gold_f1" if select_by_f1 else "val_pseudo_loss",
                    "epoch": epoch,
                    "seed": args.seed,
                },
            }, args.checkpoint)
            print(f"  -> saved best checkpoint to {args.checkpoint}")

    log_path = os.path.join(PATHS.results_dir, "stage1_train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"Training complete. Selected by {'val gold F1' if select_by_f1 else 'val pseudo loss'}. "
          f"Log: {log_path}")


if __name__ == "__main__":
    main()
