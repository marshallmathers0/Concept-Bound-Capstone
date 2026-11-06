"""
Map gold answer spans to segment ids of the CURRENT index.

Questions store their gold answer as (video, start, end) in seconds, not as
segment ids, because ids change whenever the lectures are re-segmented. A
segment counts as gold if it overlaps the gold span by at least half of the
segment's length OR covers at least half of the span (so both a long answer
spread over several short segments and a short answer inside one long
segment are handled).
"""

import json
from typing import Dict, List, Set


def gold_segments(q: Dict, metadata: List[Dict], min_frac: float = 0.5) -> Set[str]:
    out = set()
    for m in metadata:
        if m["video_id"] != q["video_id"]:
            continue
        inter = max(0.0, min(m["end_time"], q["end"]) - max(m["start_time"], q["start"]))
        if inter <= 0:
            continue
        seg_len = max(1e-6, m["end_time"] - m["start_time"])
        span_len = max(1e-6, q["end"] - q["start"])
        if inter / seg_len >= min_frac or inter / span_len >= min_frac:
            out.add(m["segment_id"])
    return out


def load_questions(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dev_test_split(q: Dict) -> str:
    """Deterministic split for threshold calibration: about half of the
    drafted questions are 'dev' (used to set the refusal threshold), the rest
    'test'. Human pilot questions are always test."""
    if q.get("source") == "pilot_human":
        return "test"
    return "dev" if sum(map(ord, q["id"])) % 2 == 0 else "test"
