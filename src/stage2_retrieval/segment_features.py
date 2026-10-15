"""
Aggregate per-time-step features (from `data/features/<video_id>.npz`) into
per-segment features (using the segment boundaries produced by Stage 1, in
`data/segments/<video_id>.json`):

  - OCR text: the distinct slide texts shown during the segment, encoded with
    Sentence-BERT (chunked + averaged, no truncation) -> e_ocr (text_embed_dim,)
  - Transcript text: the Whisper segments spoken during the segment (see
    src/utils/segment_text.py), encoded the same way -> e_transcript
  - Visual: mean-pool the CLIP embeddings of the time-steps in the segment
    (a cheap stand-in for "representative keyframe(s)") -> e_visual (clip_dim,)

The resulting per-segment feature bundle is saved to
`data/segments/<video_id>_segment_features.npz` and is consumed by both
`index_builder.py` (to build the searchable FAISS index) and
`dataset.py` (to build (query, segment) training pairs).
"""

import os
from typing import Dict, List

import numpy as np

from config import FEATURES, PATHS
from src.feature_extraction.feature_pipeline import project_clip_embeddings
from src.utils.io_utils import load_features, load_json, save_features
from src.utils.segment_text import span_ocr, span_transcript


def segment_bundle(feats: Dict, segments: List[Dict], text_encoder) -> Dict:
    """Texts + (ocr, transcript, visual) embeddings for arbitrary segments of
    one video. Shared by Stage 2 training/indexing and by the evaluation
    harness, so every segmentation is represented identically."""
    timestamps = feats["timestamps"]
    clip_embeds = project_clip_embeddings(feats["clip_embeds"], FEATURES.clip_dim)

    ocr_agg, transcript_agg, visual_agg = [], [], []
    starts, ends = [], []
    for seg in segments:
        st, et = float(seg["start_time"]), float(seg["end_time"])
        idx = np.where((timestamps >= st) & (timestamps < et))[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmin(np.abs(timestamps - st)))])
        ocr_agg.append(span_ocr(feats, st, et))
        transcript_agg.append(span_transcript(feats, st, et))
        visual_agg.append(clip_embeds[idx].mean(axis=0))
        starts.append(st)
        ends.append(et)

    return dict(
        start_times=np.array(starts, dtype=np.float32),
        end_times=np.array(ends, dtype=np.float32),
        ocr_emb=text_encoder.encode_long(ocr_agg),
        transcript_emb=text_encoder.encode_long(transcript_agg),
        visual_emb=np.stack(visual_agg, axis=0) if visual_agg else np.zeros((0, clip_embeds.shape[1]), dtype=np.float32),
        ocr_text=np.array(ocr_agg, dtype=object),
        transcript_text=np.array(transcript_agg, dtype=object),
    )


def compute_segment_features(
    video_id: str,
    text_encoder,
    features_dir: str = PATHS.features_dir,
    segments_dir: str = PATHS.segments_dir,
) -> Dict:
    feats = load_features(os.path.join(features_dir, f"{video_id}.npz"))
    seg_data = load_json(os.path.join(segments_dir, f"{video_id}.json"))
    segments: List[Dict] = seg_data["segments"]
    bundle = segment_bundle(feats, segments, text_encoder)

    result = dict(video_id=video_id, **bundle)

    out_path = os.path.join(segments_dir, f"{video_id}_segment_features.npz")
    save_features(out_path, **result)
    return result


def load_segment_features(video_id: str, segments_dir: str = PATHS.segments_dir) -> Dict:
    feats = load_features(os.path.join(segments_dir, f"{video_id}_segment_features.npz"))
    if feats["visual_emb"].shape[1] != FEATURES.clip_dim:
        feats["visual_emb"] = project_clip_embeddings(feats["visual_emb"], FEATURES.clip_dim)
    return feats
