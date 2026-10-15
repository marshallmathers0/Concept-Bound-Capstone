"""
Re-run OCR and/or ASR on existing feature files without redoing CLIP.

Use this after fixing an extractor (e.g. the earlier silent EasyOCR failure
that left `ocr_signal` all zeros, or to add Whisper hallucination filtering and
raw ASR segments). The fused `features` matrix is rebuilt from the refreshed
signals, so Stage 1 must be retrained afterwards.

Usage
-----
    python scripts/refresh_features.py --ocr --asr
    python scripts/refresh_features.py --ocr --videos college_lec01 college_lec03
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import FEATURES, MODELS, PATHS, ensure_dirs
from src.feature_extraction.asr_extractor import (
    SentenceEncoder, asr_segments_to_array, extract_topic_drift_signal,
)
from src.feature_extraction.feature_pipeline import (
    project_clip_embeddings, run_asr, run_ocr, sample_frames,
)
from src.utils.io_utils import load_features, save_features

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def find_video(video_id: str) -> str:
    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(PATHS.raw_videos_dir, video_id + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No raw video for {video_id} in {PATHS.raw_videos_dir}")


def refresh(video_id: str, do_ocr: bool, do_asr: bool, device: str):
    path = os.path.join(PATHS.features_dir, f"{video_id}.npz")
    feats = load_features(path)
    provenance = json.loads(str(feats["provenance"])) if "provenance" in feats else {}
    video_path = find_video(video_id)

    if do_ocr:
        t0 = time.time()
        frames, timestamps, _ = sample_frames(video_path, FEATURES.time_step_sec)
        if len(timestamps) != len(feats["timestamps"]):
            raise RuntimeError(f"{video_id}: {len(timestamps)} sampled frames but feature file "
                               f"has {len(feats['timestamps'])} steps; re-run extract_features.py")
        signal, texts, bboxes, ocr_prov = run_ocr(video_id, frames, device)
        del frames
        feats["ocr_signal"] = signal
        feats["ocr_texts"] = np.array(texts, dtype=object)
        feats["ocr_bboxes"] = bboxes
        provenance.update(ocr_prov)
        print(f"  [{video_id}] OCR refreshed in {time.time() - t0:.0f}s")

    if do_asr:
        t0 = time.time()
        transcript = run_asr(video_id, video_path, device)
        sbert = SentenceEncoder(MODELS.sentence_bert, device=device)
        drift, windows = extract_topic_drift_signal(transcript, feats["timestamps"], sbert, use_hybrid=True)
        feats["topic_drift_signal"] = drift
        feats["transcript_texts"] = np.array(windows, dtype=object)
        feats["asr_segments"] = asr_segments_to_array(transcript)
        provenance["whisper_model"] = MODELS.whisper_model
        provenance["whisper_filtering"] = "condition_on_previous_text=False + hallucination filter"
        print(f"  [{video_id}] ASR refreshed in {time.time() - t0:.0f}s")

    clip = project_clip_embeddings(feats["clip_embeds"], FEATURES.clip_dim)
    scalars = np.stack([feats["ocr_signal"], feats["topic_drift_signal"], feats["visual_signal"]], axis=1)
    feats["clip_embeds"] = clip
    feats["features"] = np.concatenate([scalars, clip], axis=1).astype(np.float32)
    feats["provenance"] = np.array(json.dumps(provenance))
    save_features(path, **feats)
    print(f"  [{video_id}] saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true", help="Re-run OCR (screen-cropped).")
    parser.add_argument("--asr", action="store_true", help="Re-run Whisper with hallucination filtering.")
    parser.add_argument("--videos", nargs="*", default=None, help="Video ids (default: all).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not (args.ocr or args.asr):
        parser.error("pass --ocr and/or --asr")

    ensure_dirs()
    video_ids = args.videos or [os.path.splitext(os.path.basename(p))[0]
                                for p in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))]
    for vid in video_ids:
        print(f"Refreshing {vid} (ocr={args.ocr}, asr={args.asr}) ...")
        refresh(vid, args.ocr, args.asr, args.device)
    print("Done. Retrain Stage 1 and rebuild segments, segment features and the index.")


if __name__ == "__main__":
    main()
