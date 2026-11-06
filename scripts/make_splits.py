"""
Create data/splits.json: a video-level train/val/test split.

Splitting must happen at the video (or course) level. Splitting individual
segments or queries at random puts neighbouring moments of the same lecture
in both train and test, which inflates every metric.

Freeze the test split before looking at any results, and do not tune
thresholds or hyperparameters on it.

Usage
-----
    python scripts/make_splits.py
    python scripts/make_splits.py --groups data/video_meta.json   # {"<video_id>": {"course": "..."}}
    python scripts/make_splits.py --force                          # overwrite an existing split
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from src.utils.gold import make_splits
from src.utils.io_utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--groups", default=None,
                        help="JSON {video_id: {'course': ...}} to split by course instead of by video.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(PATHS.splits_path) and not args.force:
        print(f"{PATHS.splits_path} already exists (use --force to overwrite). "
              f"Changing the test split after seeing results invalidates them.")
        return

    video_ids = [os.path.splitext(os.path.basename(p))[0]
                 for p in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))]
    groups = None
    if args.groups:
        meta = load_json(args.groups)
        groups = {v: meta[v]["course"] for v in video_ids}

    split = make_splits(video_ids, args.test_frac, args.val_frac, args.seed, groups)
    save_json(split, PATHS.splits_path)
    for k in ("train", "val", "test"):
        print(f"{k:5s} ({len(split[k])}): {split[k]}")


if __name__ == "__main__":
    main()
