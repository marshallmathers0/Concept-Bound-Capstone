"""
Build the persistent FAISS index over course-corpus segment embeddings.

For every video with precomputed segment features
(`data/segments/<video_id>_segment_features.npz`), this script:
  1. Loads the trained CrossModalRetrievalModel (segment fusion encoder).
  2. Computes z_seg for every segment.
  3. Appends the (L2-normalized) embeddings to a FAISS `IndexFlatIP` index
     (inner product on normalized vectors == cosine similarity).
  4. Writes a parallel metadata JSON list (video_id, start_time, end_time,
     ocr_text, transcript_text) so that search results can be mapped back to
     human-readable "jump to moment" links.

Usage
-----
    python -m src.stage2_retrieval.index_builder \
        --segments-dir data/segments \
        --checkpoint checkpoints/stage2_retrieval_model.pt \
        --index-dir data/index
"""

import argparse
import glob
import os

import numpy as np
import torch

from config import PATHS, ensure_dirs
from src.stage2_retrieval.fusion_model import CrossModalRetrievalModel
from src.utils.io_utils import save_json


def load_stage2_model(checkpoint_path: str, device: str = "cpu") -> CrossModalRetrievalModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CrossModalRetrievalModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def list_segment_feature_video_ids(segments_dir: str):
    paths = sorted(glob.glob(os.path.join(segments_dir, "*_segment_features.npz")))
    return [os.path.basename(p).replace("_segment_features.npz", "") for p in paths]


def build_index(segments_dir: str, checkpoint_path: str, index_dir: str, device: str = "cpu"):
    import faiss
    from src.stage2_retrieval.segment_features import load_segment_features

    ensure_dirs()
    os.makedirs(index_dir, exist_ok=True)

    model = load_stage2_model(checkpoint_path, device)
    out_dim = model.segment_encoder.out_proj[-1].out_features

    index = faiss.IndexFlatIP(out_dim)
    metadata = []

    video_ids = list_segment_feature_video_ids(segments_dir)
    if not video_ids:
        raise RuntimeError(f"No '*_segment_features.npz' files found in {segments_dir}.")

    for video_id in video_ids:
        feats = load_segment_features(video_id, segments_dir)
        n_segments = len(feats["start_times"])
        if n_segments == 0:
            continue

        ocr_emb = torch.from_numpy(feats["ocr_emb"].astype(np.float32)).to(device)
        transcript_emb = torch.from_numpy(feats["transcript_emb"].astype(np.float32)).to(device)
        visual_emb = torch.from_numpy(feats["visual_emb"].astype(np.float32)).to(device)

        with torch.no_grad():
            z_seg = model.encode_segments(ocr_emb, transcript_emb, visual_emb)
        z_seg_np = z_seg.cpu().numpy().astype(np.float32)

        index.add(z_seg_np)

        for i in range(n_segments):
            metadata.append({
                "video_id": video_id,
                "start_time": float(feats["start_times"][i]),
                "end_time": float(feats["end_times"][i]),
                "ocr_text": str(feats["ocr_text"][i])[:300],
                "transcript_text": str(feats["transcript_text"][i])[:300],
            })

    faiss.write_index(index, os.path.join(index_dir, "segments.index"))
    save_json(metadata, os.path.join(index_dir, "segments_metadata.json"))
    save_json({"out_dim": out_dim, "num_segments": len(metadata)}, os.path.join(index_dir, "index_info.json"))

    print(f"Indexed {len(metadata)} segments from {len(video_ids)} videos -> {index_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    parser.add_argument("--index-dir", default=PATHS.index_dir)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    build_index(args.segments_dir, args.checkpoint, args.index_dir, args.device)


if __name__ == "__main__":
    main()
