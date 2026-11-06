"""
PyTorch Dataset for Stage-1 boundary-detector training.

Each item corresponds to one lecture video's feature stream
(`data/features/<video_id>.npz`, produced by
`src.feature_extraction.feature_pipeline`).

Where the boundary labels come from is controlled by `label_source`:

  - "gold":   human boundaries from data/annotations/gold/ only. Videos
              without gold are skipped.
  - "pseudo": peaks of the OCR / topic-drift / visual signals for every video.
              These are derived from the model's own input features, so a
              model trained only on them can at best re-learn the heuristic.
              Never report pseudo-label agreement as accuracy.
  - "auto":   gold where a video has it, pseudo elsewhere (default).

For videos with gold annotations the sequence is cropped to the annotated
`lecture_span`, so dead air before/after the lecture is not labelled as one
giant segment.

Because lecture videos vary in length, the dataset returns full sequences and
a custom `collate_fn` pads them to the same length within a batch (with a
padding mask passed to the Transformer encoder).
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import PATHS
from src.feature_extraction.feature_pipeline import load_feature_stream
from src.utils.gold import load_gold
from src.stage1_boundary.pseudo_labels import (
    pseudo_boundaries_from_signals,
    boundaries_from_annotations,
    boundaries_to_segment_ids,
)

LABEL_SOURCES = ("auto", "gold", "pseudo")


def span_mask(timestamps: np.ndarray, gold: Optional[dict]) -> np.ndarray:
    """Boolean (T,) mask of time-steps inside the gold `lecture_span`."""
    if gold is None or not gold.get("lecture_span"):
        return np.ones(len(timestamps), dtype=bool)
    lo, hi = gold["lecture_span"]
    return (timestamps >= lo) & (timestamps < hi)


def labels_for_video(feats: dict, gold: Optional[dict], label_source: str):
    """Return (keep_mask, is_boundary over kept steps, source_used)."""
    timestamps = feats["timestamps"].astype(np.float32)
    # A gold file may hold only queries (e.g. a pilot without boundary
    # annotation); that is not a claim that the lecture has no boundaries.
    if gold is not None and not gold.get("boundaries_sec"):
        gold = None
    if label_source in ("auto", "gold") and gold is not None:
        keep = span_mask(timestamps, gold)
        is_boundary = boundaries_from_annotations(timestamps[keep], gold["boundaries_sec"])
        return keep, is_boundary, "gold"
    if label_source == "gold":
        return None, None, None
    keep = np.ones(len(timestamps), dtype=bool)
    is_boundary = pseudo_boundaries_from_signals(
        feats["ocr_signal"].astype(np.float32),
        feats["topic_drift_signal"].astype(np.float32),
        feats["visual_signal"].astype(np.float32),
    )
    return keep, is_boundary, "pseudo"


class BoundaryDataset(Dataset):
    def __init__(self, video_ids: List[str], features_dir: str = PATHS.features_dir,
                 label_source: str = "auto"):
        if label_source not in LABEL_SOURCES:
            raise ValueError(f"label_source must be one of {LABEL_SOURCES}")
        self.features_dir = features_dir
        self.label_source = label_source
        self.items = []
        self.sources = {}
        for video_id in video_ids:
            feats = load_feature_stream(os.path.join(features_dir, f"{video_id}.npz"))
            keep, is_boundary, used = labels_for_video(feats, load_gold(video_id), label_source)
            if used is None:
                continue
            self.items.append((video_id, feats["features"][keep], feats["timestamps"][keep], is_boundary))
            self.sources[video_id] = used

    @property
    def video_ids(self):
        return [it[0] for it in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_id, features, timestamps, is_boundary = self.items[idx]
        seg_ids = boundaries_to_segment_ids(is_boundary)
        return {
            "video_id": video_id,
            "features": torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            "is_boundary": torch.from_numpy(is_boundary).long(),
            "seg_ids": torch.from_numpy(seg_ids).long(),
            "timestamps": torch.from_numpy(timestamps.astype(np.float32)),
        }


def collate_fn(batch):
    max_T = max(item["features"].shape[0] for item in batch)
    D = batch[0]["features"].shape[1]
    B = len(batch)

    features = torch.zeros(B, max_T, D)
    is_boundary = torch.zeros(B, max_T - 1, dtype=torch.long)
    seg_ids = torch.zeros(B, max_T, dtype=torch.long)
    key_padding_mask = torch.ones(B, max_T, dtype=torch.bool)  # True = pad
    lengths = torch.zeros(B, dtype=torch.long)
    video_ids = []

    for i, item in enumerate(batch):
        T = item["features"].shape[0]
        features[i, :T] = item["features"]
        is_boundary[i, : T - 1] = item["is_boundary"]
        # Pad seg_ids with a unique segment id so padded steps never count as
        # "same segment" with real ones during InfoNCE sampling.
        seg_ids[i, :T] = item["seg_ids"]
        if T < max_T:
            seg_ids[i, T:] = seg_ids[i, :T].max() + 100
        key_padding_mask[i, :T] = False
        lengths[i] = T
        video_ids.append(item["video_id"])

    return {
        "video_ids": video_ids,
        "features": features,
        "is_boundary": is_boundary,
        "seg_ids": seg_ids,
        "key_padding_mask": key_padding_mask,
        "lengths": lengths,
    }
