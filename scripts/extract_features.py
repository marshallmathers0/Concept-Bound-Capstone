"""
Run the full feature-extraction pipeline (OCR + ASR + visual-change + CLIP)
over every video in `data/raw_videos/` and save the fused per-time-step
feature streams to `data/features/`.

Requires: opencv-python, ffmpeg (system binary), openai-whisper, easyocr (or
tesseract), sentence-transformers, transformers (for CLIP). See
requirements.txt.

Usage
-----
    python scripts/extract_features.py
    python scripts/extract_features.py --video path/to/lecture.mp4
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import PATHS, ensure_dirs
from src.feature_extraction.feature_pipeline import build_feature_stream

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-videos-dir", default=PATHS.raw_videos_dir)
    parser.add_argument("--video", default=None, help="Process a single video file instead of the whole directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run GPU-accelerated models on. Use cuda or cpu.")
    parser.add_argument("--sequential-gpu", action="store_true",
                        help="Load models one at a time and free GPU memory between them. Prevents CUDA OOM.")
    parser.add_argument("--save-pt", action="store_true",
                        help="Additionally save features as .pt tensor files for offline caching.")
    args = parser.parse_args()

    ensure_dirs()

    if args.video:
        video_paths = [args.video]
    else:
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(glob.glob(os.path.join(args.raw_videos_dir, f"*{ext}")))
        video_paths = sorted(video_paths)

    if not video_paths:
        print(f"No videos found in {args.raw_videos_dir}. "
              f"Place .mp4/.mkv/.avi/.mov/.webm files there, or pass --video.")
        return

    for video_path in video_paths:
        print(f"Extracting features from {video_path} ...")
        result = build_feature_stream(video_path, save=True, device=args.device,
                                       sequential_gpu=args.sequential_gpu,
                                       save_as_pt=args.save_pt)
        print(f"  -> {result['video_id']}.npz  (T={len(result['timestamps'])} time-steps, "
              f"duration={float(result['duration'][0]):.1f}s)")


if __name__ == "__main__":
    main()
