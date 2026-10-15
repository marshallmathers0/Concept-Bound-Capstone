"""
Human gold annotations and video-level data splits.

Gold files live in `data/annotations/gold/<video_id>.<annotator>.json`:

    {
      "video_id": "college_lec03",
      "annotator": "A1",
      "lecture_span": [95.0, 3790.0],          # teaching actually happening
      "boundaries_sec": [412.0, 905.5, ...],   # concept boundaries
      "queries": [
        {"query": "what does a language model output",
         "start_time": 120.0, "end_time": 260.0,
         "concept": "language_model_definition",
         "type": "definition"}
      ]
    }

See docs/ANNOTATION_GUIDELINES.md for how to fill them in. Several annotators
can label the same video (one file each); boundaries from the *primary*
annotator (alphabetically first, unless `primary` is given) are the evaluation
target, the others are used for inter-annotator agreement. Queries from all
annotators are pooled.

Pseudo-labels and pseudo-queries are NEVER read from here: anything in this
folder is treated as human ground truth.
"""

import glob
import json
import os
import random
from typing import Dict, List, Optional

from config import PATHS
from src.utils.io_utils import load_json, save_json

QUERY_TYPES = ("definition", "explanation", "example", "derivation",
               "comparison", "application", "procedure", "other")


def gold_files(video_id: str, gold_dir: str = PATHS.gold_dir) -> Dict[str, str]:
    """{annotator: path} for one video."""
    out = {}
    for path in sorted(glob.glob(os.path.join(gold_dir, f"{video_id}.*.json"))):
        annotator = os.path.basename(path)[len(video_id) + 1:-len(".json")]
        if annotator and "." not in annotator:
            out[annotator] = path
    return out


def list_gold_videos(gold_dir: str = PATHS.gold_dir) -> List[str]:
    vids = set()
    for path in glob.glob(os.path.join(gold_dir, "*.*.json")):
        vids.add(os.path.basename(path).split(".")[0])
    return sorted(vids)


def load_gold(video_id: str, primary: Optional[str] = None,
              gold_dir: str = PATHS.gold_dir) -> Optional[Dict]:
    """Merged gold annotation for a video, or None if it has none.

    Returns {video_id, annotators, lecture_span, boundaries_sec (primary),
    boundaries_by_annotator, queries (pooled, each tagged with annotator)}.
    """
    files = gold_files(video_id, gold_dir)
    if not files:
        return None
    anns = {a: load_json(p) for a, p in files.items()}
    primary = primary if primary in anns else sorted(anns)[0]
    queries = []
    for a, ann in anns.items():
        for q in ann.get("queries", []):
            queries.append({**q, "annotator": a})
    p = anns[primary]
    return {
        "video_id": video_id,
        "annotators": sorted(anns),
        "primary": primary,
        "lecture_span": p.get("lecture_span"),
        "boundaries_sec": sorted(float(b) for b in p.get("boundaries_sec", [])),
        "boundaries_by_annotator": {a: sorted(float(b) for b in ann.get("boundaries_sec", []))
                                    for a, ann in anns.items()},
        "queries": queries,
    }


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_splits(video_ids: List[str], test_frac: float = 0.2, val_frac: float = 0.15,
                seed: int = 13, groups: Optional[Dict[str, str]] = None) -> Dict:
    """Deterministic video-level split. With `groups` ({video_id: course}),
    whole courses are assigned to one split so the test set is a course the
    model has never seen."""
    rng = random.Random(seed)
    units = sorted(set(groups[v] for v in video_ids)) if groups else sorted(video_ids)
    rng.shuffle(units)
    n_test = max(1, round(len(units) * test_frac))
    n_val = max(1, round(len(units) * val_frac)) if len(units) - n_test > 1 else 0
    test_u, val_u = set(units[:n_test]), set(units[n_test:n_test + n_val])

    def unit(v):
        return groups[v] if groups else v

    split = {"train": [], "val": [], "test": []}
    for v in sorted(video_ids):
        key = "test" if unit(v) in test_u else "val" if unit(v) in val_u else "train"
        split[key].append(v)
    split["seed"] = seed
    split["grouped_by"] = "course" if groups else "video"
    return split


def load_splits(all_video_ids: List[str], path: str = PATHS.splits_path,
                create_if_missing: bool = True) -> Dict:
    """Load data/splits.json. Videos not listed in it (e.g. newly added
    lectures) are reported and treated as train-only, never test."""
    if not os.path.exists(path):
        if not create_if_missing:
            raise FileNotFoundError(f"{path} not found; run scripts/make_splits.py")
        split = make_splits(all_video_ids)
        save_json(split, path)
        print(f"[splits] Created {path}: " + ", ".join(f"{k}={len(split[k])}" for k in ("train", "val", "test")))
        return split
    split = load_json(path)
    listed = set(split["train"]) | set(split["val"]) | set(split["test"])
    unlisted = sorted(set(all_video_ids) - listed)
    if unlisted:
        print(f"[splits] {len(unlisted)} video(s) not in {os.path.basename(path)}, used for training only: {unlisted}. "
              f"Re-run scripts/make_splits.py to re-balance.")
        split = {**split, "train": sorted(split["train"] + unlisted)}
    return split
