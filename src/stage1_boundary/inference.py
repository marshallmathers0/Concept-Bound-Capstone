"""
Stage-1 inference: load a trained PedagogicalBoundaryDetector checkpoint and
convert a video's feature stream into concept-coherent segments.
"""

import os
from typing import Dict, List

import numpy as np
import torch

from config import PATHS, STAGE1
from src.feature_extraction.feature_pipeline import load_feature_stream
from src.stage1_boundary.model import PedagogicalBoundaryDetector
from src.stage1_boundary.segment import segments_from_scores
from src.utils.io_utils import save_json, segments_path_for


def load_stage1_model(checkpoint_path: str, device: str = "cpu") -> PedagogicalBoundaryDetector:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PedagogicalBoundaryDetector(**cfg).to(device)
    state = ckpt["model_state_dict"]
    if "input_mask" not in state:  # checkpoints from before the ablation flag
        state["input_mask"] = model.input_mask
    model.load_state_dict(state)
    model.eval()
    return model


def segment_video(
    video_id: str,
    model: PedagogicalBoundaryDetector,
    features_dir: str = PATHS.features_dir,
    device: str = "cpu",
    threshold: float = None,
    keep_mask: np.ndarray = None,
) -> List[Dict]:
    """Run the boundary detector on one video's features and return a list
    of concept-coherent segment dicts: {start_idx, end_idx, start_time, end_time}.

    `keep_mask` (T,) restricts segmentation to part of the video (e.g. the
    gold `lecture_span` during evaluation); indices are then relative to it."""
    feats = load_feature_stream(os.path.join(features_dir, f"{video_id}.npz"))
    keep = keep_mask if keep_mask is not None else np.ones(len(feats["timestamps"]), dtype=bool)
    features = torch.from_numpy(np.ascontiguousarray(feats["features"][keep])).unsqueeze(0).to(device)
    timestamps = feats["timestamps"][keep].astype(np.float32)

    with torch.no_grad():
        z = model(features)  # (1, T, E)
        scores = model.boundary_scores(z)[0].cpu().numpy()  # (T-1,)

    segments = segments_from_scores(scores, timestamps, threshold=threshold)
    return segments


def segment_and_save(video_id: str, model: PedagogicalBoundaryDetector,
                      features_dir: str = PATHS.features_dir,
                      segments_dir: str = PATHS.segments_dir,
                      device: str = "cpu",
                      threshold: float = None) -> List[Dict]:
    segments = segment_video(video_id, model, features_dir, device, threshold=threshold)
    save_json({"video_id": video_id, "segments": segments}, segments_path_for(video_id, segments_dir))
    return segments
