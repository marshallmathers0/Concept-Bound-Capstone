"""
Generic I/O helpers shared across the pipeline:
 - JSON / NPZ load-save wrappers
 - Video metadata helpers (id derivation, output path conventions)
"""

import json
import os
import numpy as np


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_features(path: str, **arrays):
    """Save a dict of numpy arrays (and lists of strings) to a compressed npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_features(path: str):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def video_id_from_path(video_path: str) -> str:
    """Derive a stable video_id from a file path (filename without extension)."""
    base = os.path.basename(video_path)
    return os.path.splitext(base)[0]


def feature_path_for(video_id: str, features_dir: str) -> str:
    return os.path.join(features_dir, f"{video_id}.npz")


def segments_path_for(video_id: str, segments_dir: str) -> str:
    return os.path.join(segments_dir, f"{video_id}.json")


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for human-readable output."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_pt(path: str, data: dict):
    """Save a dict of tensors/data to a PyTorch .pt file for offline caching.

    This decouples heavy feature extraction (GPU-intensive) from model training,
    eliminating CUDA Out-of-Memory crashes when both run simultaneously.
    """
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def load_pt(path: str) -> dict:
    """Load a cached PyTorch .pt feature file."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)

