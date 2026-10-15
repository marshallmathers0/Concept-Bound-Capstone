"""
Training loop for the Cross-Modal Segment Fusion + Retrieval model (Stage 2).

Trains the SegmentFusionEncoder and QueryProjector jointly with a symmetric
InfoNCE loss over (query, segment) pairs, using in-batch negatives.

Usage
-----
    python -m src.stage2_retrieval.train \
        --annotations-dir data/annotations \
        --segments-dir data/segments \
        --checkpoint checkpoints/stage2_retrieval_model.pt \
        --fusion-mode cross_attention
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import STAGE2, PATHS, ensure_dirs
from src.stage2_retrieval.dataset import RetrievalPairDataset
from src.utils.gold import load_splits
from src.stage2_retrieval.encoders import TextEncoder
from src.stage2_retrieval.fusion_model import CrossModalRetrievalModel


def info_nce_loss(query_z: torch.Tensor, seg_z: torch.Tensor, temperature: float,
                  seg_key: torch.Tensor = None) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of (query, segment) pairs using
    in-batch negatives.

    query_z, seg_z: (B, D), both L2-normalized.
    seg_key: optional (B,) target-segment ids. Off-diagonal pairs that share
    a target segment are masked out: they are positives, not negatives
    (false negatives otherwise push identical segments apart).
    """
    logits = query_z @ seg_z.t() / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    if seg_key is not None:
        same = seg_key.unsqueeze(0) == seg_key.unsqueeze(1)
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))

    loss_q2s = F.cross_entropy(logits, labels)
    loss_s2q = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q2s + loss_s2q)


def run_epoch(model, loader, optimizer, device, temperature, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        ocr_emb = batch["ocr_emb"].to(device)
        transcript_emb = batch["transcript_emb"].to(device)
        visual_emb = batch["visual_emb"].to(device)
        query_emb = batch["query_emb"].to(device)
        seg_key = batch["seg_key"].to(device) if "seg_key" in batch else None

        with torch.set_grad_enabled(train):
            seg_z = model.encode_segments(ocr_emb, transcript_emb, visual_emb)
            query_z = model.encode_queries(query_emb)
            loss = info_nce_loss(query_z, seg_z, temperature, seg_key)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", default=PATHS.annotations_dir)
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    parser.add_argument("--epochs", type=int, default=STAGE2.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE2.batch_size)
    parser.add_argument("--lr", type=float, default=STAGE2.lr)
    parser.add_argument("--temperature", type=float, default=STAGE2.temperature)
    parser.add_argument("--fusion-mode", default="cross_attention", choices=["cross_attention", "concat"])
    parser.add_argument("--use-gold", action="store_true",
                        help="Also train on human gold queries of TRAIN-split videos.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Video-level split: queries from one lecture never appear in both train
    # and val (a random split over segments leaks neighbouring content).
    all_ids = [os.path.basename(p).replace("_segment_features.npz", "")
               for p in sorted(glob.glob(os.path.join(args.segments_dir, "*_segment_features.npz")))]
    split = load_splits(all_ids)

    text_encoder = TextEncoder(device=args.device)
    train_ds = RetrievalPairDataset(split["train"], text_encoder, args.annotations_dir, args.segments_dir,
                                    use_gold=args.use_gold)
    # Validate on pseudo-queries of val videos (gold queries of val/test are
    # reserved for evaluation).
    val_ds = RetrievalPairDataset(split["val"], text_encoder, args.annotations_dir, args.segments_dir)
    if len(train_ds) == 0:
        raise RuntimeError("No (query, segment) training pairs. Run generate_pseudo_queries.py "
                           "(and prepare_segment_features.py) first.")
    print(f"Train pairs: {len(train_ds)} {train_ds.counts} | Val pairs: {len(val_ds)} {val_ds.counts}")
    n_train = len(train_ds)
    if len(val_ds) == 0:
        val_ds = None
    batch_size = min(args.batch_size, max(2, n_train))
    pin_memory = args.device.startswith('cuda') if isinstance(args.device, str) else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(n_train > batch_size), pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if val_ds else None

    # Infer dimensionalities from the actual precomputed segment features
    # (Sentence-BERT dim for text, CLIP dim for visual) rather than assuming
    # config.py defaults. This keeps the training script robust to real data.
    sample = train_ds[0]
    ocr_dim = sample["ocr_emb"].shape[0]
    transcript_dim = sample["transcript_emb"].shape[0]
    visual_dim = sample["visual_emb"].shape[0]
    query_dim = sample["query_emb"].shape[0]
    print(f"Detected dims: ocr={ocr_dim} transcript={transcript_dim} visual={visual_dim} query={query_dim}")

    model = CrossModalRetrievalModel(
        ocr_dim=ocr_dim,
        transcript_dim=transcript_dim,
        visual_dim=visual_dim,
        query_dim=query_dim,
        hidden_dim=STAGE2.fusion_hidden_dim,
        out_dim=STAGE2.fusion_out_dim,
        num_heads=STAGE2.num_heads,
        dropout=STAGE2.dropout,
        fusion_mode=args.fusion_mode,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, args.temperature, train=True)
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, args.device, args.temperature, train=False)
        else:
            val_loss = train_loss

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "ocr_dim": ocr_dim,
                    "transcript_dim": transcript_dim,
                    "visual_dim": visual_dim,
                    "query_dim": query_dim,
                    "hidden_dim": STAGE2.fusion_hidden_dim,
                    "out_dim": STAGE2.fusion_out_dim,
                    "num_heads": STAGE2.num_heads,
                    "dropout": STAGE2.dropout,
                    "fusion_mode": args.fusion_mode,
                },
                "train_meta": {"pairs": train_ds.counts, "use_gold": args.use_gold,
                               "epoch": epoch, "seed": args.seed},
            }, args.checkpoint)
            print(f"  -> saved best checkpoint to {args.checkpoint} (val_loss={val_loss:.4f})")

    log_path = os.path.join(PATHS.results_dir, f"stage2_train_log_{args.fusion_mode}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"Training complete. Best val loss: {best_val:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
