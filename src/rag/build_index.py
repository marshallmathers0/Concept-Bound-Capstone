"""
Offline indexing for the RAG demo: videos -> segments -> embeddings -> artifacts.

    python -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05
    python -m src.rag.build_index --videos path/to/videos --limit 5 --extract-missing

For each video:
  1. features: reuse data/features/<id>.npz; if missing and --extract-missing,
     run the full extractor (Whisper + OCR + CLIP; slow, offline only)
  2. Stage 1 boundary detector -> concept-coherent segments
  3. segment texts (transcript + distinct slide text) and the three modality
     embeddings; Stage 2 fusion encoder -> z_seg (FAISS, inner product)
  4. Sentence-BERT embeddings of each segment's text (cached so the app does
     not re-encode at startup) - the dense and hybrid retrievers use these

Artifacts in --out (default data/rag_index/), loaded by LectureRetriever:
  segments.index           FAISS IndexFlatIP over fusion embeddings
  segments_metadata.json   one record per segment (id, video, title, url, times, texts)
  text_embeddings.npy      Sentence-BERT embeddings, aligned with the metadata
  index_info.json          what was built, from which checkpoints, when

The demo app and the eval only read these files; they never run Whisper/OCR.
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from config import MODELS, PATHS, ensure_dirs
from src.utils.io_utils import format_timestamp, load_features, load_json, save_json

DEFAULT_OUT = os.path.join(os.path.dirname(PATHS.index_dir), "rag_index")
VIDEO_META = os.path.join(os.path.dirname(PATHS.index_dir), "video_meta.json")
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def segment_id(video_id: str, i: int) -> str:
    return f"{video_id}#{i:03d}"


def find_video_file(video_id: str, videos_dir: str):
    for ext in VIDEO_EXTS:
        p = os.path.join(videos_dir, video_id + ext)
        if os.path.exists(p):
            return p
    return None


def ensure_features(video_id: str, videos_dir: str, extract_missing: bool, device: str) -> str:
    path = os.path.join(PATHS.features_dir, f"{video_id}.npz")
    if os.path.exists(path):
        return path
    video = find_video_file(video_id, videos_dir)
    if not (extract_missing and video):
        raise FileNotFoundError(f"{video_id}: no features at {path} (pass --extract-missing to run Whisper/OCR/CLIP)")
    from src.feature_extraction.feature_pipeline import build_feature_stream
    print(f"  [{video_id}] extracting features (slow; offline only) ...")
    build_feature_stream(video, save=True, device=device)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=PATHS.raw_videos_dir, help="Folder of lecture videos.")
    ap.add_argument("--ids", nargs="*", default=None, help="Explicit video ids (default: videos in --videos).")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--meta", default=VIDEO_META)
    ap.add_argument("--extract-missing", action="store_true")
    ap.add_argument("--stage1-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    ap.add_argument("--stage2-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ensure_dirs()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import faiss
    from src.stage1_boundary.inference import load_stage1_model, segment_video
    from src.stage2_retrieval.encoders import TextEncoder
    from src.stage2_retrieval.index_builder import load_stage2_model
    from src.stage2_retrieval.segment_features import segment_bundle

    ids = args.ids or sorted(os.path.splitext(os.path.basename(p))[0]
                             for ext in VIDEO_EXTS for p in glob.glob(os.path.join(args.videos, "*" + ext)))
    ids = ids[:args.limit]
    if not ids:
        sys.exit(f"No videos found in {args.videos}")
    meta_all = load_json(args.meta) if os.path.exists(args.meta) else {}

    stage1 = load_stage1_model(args.stage1_checkpoint, args.device)
    stage2 = load_stage2_model(args.stage2_checkpoint, args.device)
    text_enc = TextEncoder(device=args.device)

    records, fusion_z = [], []
    for vid in ids:
        ensure_features(vid, args.videos, args.extract_missing, args.device)
        feats = load_features(os.path.join(PATHS.features_dir, f"{vid}.npz"))
        if "asr_segments" not in feats:
            print(f"  [{vid}] WARNING: old-format features (no raw Whisper segments, OCR may be empty); "
                  f"run scripts/refresh_features.py --ocr --asr --videos {vid}")
        segs = segment_video(vid, stage1, PATHS.features_dir, args.device)
        b = segment_bundle(feats, segs, text_enc)
        with torch.no_grad():
            z = stage2.encode_segments(*(torch.from_numpy(b[k].astype(np.float32)).to(args.device)
                                         for k in ("ocr_emb", "transcript_emb", "visual_emb")))
        z = z.cpu().numpy().astype(np.float32)
        m = meta_all.get(vid, {})
        # Respect the annotated teaching span: drops pre-class chatter and
        # roll calls (student names) from what the demo can show or cite.
        from src.utils.gold import load_gold
        span = (load_gold(vid) or {}).get("lecture_span") or m.get("lecture_span")
        for i, seg in enumerate(segs):
            st, et = float(seg["start_time"]), float(seg["end_time"])
            if span and (et <= span[0] or st >= span[1]):
                continue
            fusion_z.append(z[i:i + 1])  # keep vectors aligned with the kept records
            records.append({
                "segment_id": segment_id(vid, i),
                "video_id": vid,
                "title": m.get("title", vid),
                "course": m.get("course", ""),
                "url": m.get("url") or f"/videos/{vid}.mp4",
                "is_youtube": "youtu" in (m.get("url") or ""),
                "license": m.get("license", ""),
                "start_time": st, "end_time": et,
                "start_str": format_timestamp(st), "end_str": format_timestamp(et),
                "transcript_text": str(b["transcript_text"][i]),
                "ocr_text": str(b["ocr_text"][i]),
                "snippet": str(b["transcript_text"][i])[:300],
            })
        kept = sum(1 for r in records if r["video_id"] == vid)
        print(f"  [{vid}] {kept} segments" + (f" ({len(segs) - kept} outside lecture span dropped)" if kept < len(segs) else ""), flush=True)

    z = np.concatenate(fusion_z, axis=0)
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    faiss.write_index(index, os.path.join(args.out, "segments.index"))
    save_json(records, os.path.join(args.out, "segments_metadata.json"))

    texts = [(r["transcript_text"] + " " + r["ocr_text"]).strip() or " " for r in records]
    emb = text_enc.encode_long(texts)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.save(os.path.join(args.out, "text_embeddings.npy"), emb.astype(np.float32))

    save_json({
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "videos": ids, "num_segments": len(records), "fusion_dim": int(z.shape[1]),
        "text_encoder": MODELS.sentence_bert,
        "stage1_checkpoint": os.path.basename(args.stage1_checkpoint),
        "stage2_checkpoint": os.path.basename(args.stage2_checkpoint),
        "build_seconds": round(time.time() - t0, 1),
    }, os.path.join(args.out, "index_info.json"))
    print(f"Built {len(records)} segments from {len(ids)} videos in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
