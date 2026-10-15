"""
One-command pipeline runner: raw videos -> features -> Stage 1 -> segments ->
segment features -> pseudo-queries -> Stage 2 -> FAISS index.

Each step calls the same script you would run by hand, so there is exactly
one implementation of every step (an earlier version duplicated the Stage-2
training loop here, with a leaky random split).

Usage
-----
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --force          # re-do every step
    python scripts/run_full_pipeline.py --skip-training  # reuse existing checkpoints
"""

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch

from config import PATHS, ensure_dirs


def run(cmd):
    print("  $ " + " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": ROOT}
    return subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env).returncode == 0


def missing_features():
    vids = []
    for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        vids += glob.glob(os.path.join(PATHS.raw_videos_dir, f"*{ext}"))
    return [v for v in sorted(vids)
            if not os.path.exists(os.path.join(PATHS.features_dir,
                                               os.path.splitext(os.path.basename(v))[0] + ".npz"))]


def main():
    parser = argparse.ArgumentParser(description="Run the full Lecture Retrieval pipeline.")
    parser.add_argument("--force", action="store_true", help="Re-extract features for every video.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ensure_dirs()
    dev = ["--device", args.device]
    s1_ckpt = os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt")
    s2_ckpt = os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")

    steps = []
    todo = None if args.force else missing_features()
    if args.force:
        steps.append(("Extract features (all videos)", ["scripts/extract_features.py"] + dev))
    for v in todo or []:
        steps.append((f"Extract features: {os.path.basename(v)}", ["scripts/extract_features.py", "--video", v] + dev))
    if not os.path.exists(PATHS.splits_path):
        steps.append(("Create video-level splits", ["scripts/make_splits.py"]))
    if not args.skip_training or not os.path.exists(s1_ckpt):
        steps.append(("Train Stage 1", ["-m", "src.stage1_boundary.train"] + dev))
    steps += [
        ("Segment videos", ["scripts/segment_videos.py"] + dev),
        ("Segment features", ["scripts/prepare_segment_features.py"] + dev),
        ("Pseudo-queries (training only)", ["scripts/generate_pseudo_queries.py"]),
    ]
    if not args.skip_training or not os.path.exists(s2_ckpt):
        steps.append(("Train Stage 2", ["-m", "src.stage2_retrieval.train"] + dev))
    steps.append(("Build FAISS index", ["-m", "src.stage2_retrieval.index_builder"] + dev))

    t_all = time.time()
    for i, (label, cmd) in enumerate(steps, 1):
        print(f"\n{'=' * 60}\n  {i}/{len(steps)}  {label}\n{'=' * 60}")
        t0 = time.time()
        if not run(cmd):
            print(f"  [!] '{label}' failed; stopping. Later steps depend on it.")
            sys.exit(1)
        print(f"  [TIME] {time.time() - t0:.1f}s")

    print(f"\nPipeline complete in {time.time() - t_all:.0f}s.")
    print("Evaluate (needs human gold annotations): python scripts/run_evaluation.py")
    print("Demo app: uvicorn app.backend.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
