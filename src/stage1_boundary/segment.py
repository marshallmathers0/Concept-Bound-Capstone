"""
Convert per-time-step boundary scores from the trained Pedagogical Boundary
Detector into a list of discrete, concept-coherent segments
(start_time, end_time), with light post-processing:

  - peak-pick the boundary-score sequence using the same `find_peaks`
    machinery used for pseudo-labels (but on the *learned* score, not the
    raw signals)
  - enforce plausible concept-segment lengths: boundaries closer than
    `min_segment_sec` are suppressed (the stronger one survives), and
    segments longer than `max_segment_sec` are split at their strongest
    internal score. Without this the detector produced 12-30 second
    "concepts" next to 27-minute ones.
"""

from typing import List, Dict
import numpy as np
from scipy.signal import find_peaks

from config import STAGE1


def boundary_scores_to_boundaries(
    boundary_scores: np.ndarray,
    threshold: float = None,
    min_distance: int = STAGE1.peak_min_distance_steps,
    adaptive_std: float = STAGE1.adaptive_threshold_std,
) -> np.ndarray:
    """boundary_scores: (T-1,) array in [0, 1] (output of model.boundary_scores).

    If `threshold` is None (default), an adaptive threshold is used:
        threshold = mean(boundary_scores) + adaptive_std * std(boundary_scores)
    This is more robust than a fixed absolute cosine-similarity threshold,
    since the absolute scale of "1 - cos_sim" depends on the embedding
    dimensionality and how long/strongly the model has been trained.

    Returns indices t such that a boundary lies between time-step t and t+1.
    """
    if len(boundary_scores) == 0:
        return np.array([], dtype=np.int64)

    if threshold is None:
        mu, sigma = float(boundary_scores.mean()), float(boundary_scores.std())
        threshold = mu + adaptive_std * sigma

    peaks, _ = find_peaks(boundary_scores, height=threshold, distance=min_distance)
    return peaks


def boundaries_to_segments(
    boundary_indices: np.ndarray,
    timestamps: np.ndarray,
    min_segment_len_steps: int = STAGE1.min_segment_len_steps,
) -> List[Dict]:
    """Build a list of {start_idx, end_idx, start_time, end_time} segments
    from boundary indices, merging short segments into their neighbours.
    """
    T = len(timestamps)
    if T == 0:
        return []

    cut_points = sorted(set([0] + [int(b) + 1 for b in boundary_indices] + [T]))

    # Merge adjacent segments shorter than the minimum length.
    merged = [cut_points[0]]
    for cp in cut_points[1:]:
        if cp - merged[-1] < min_segment_len_steps and cp != T:
            continue  # skip this cut point, effectively merging
        merged.append(cp)
    if merged[-1] != T:
        merged[-1] = T

    segments = []
    for i in range(len(merged) - 1):
        start_idx, end_idx = merged[i], merged[i + 1]
        if start_idx >= end_idx:
            continue
        start_time = float(timestamps[start_idx])
        # end time = timestamp of the time-step after the last one in this
        # segment (or, for the final segment, extrapolate one step).
        if end_idx < T:
            end_time = float(timestamps[end_idx])
        else:
            step = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0
            end_time = float(timestamps[-1] + step)

        segments.append({
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_time": start_time,
            "end_time": end_time,
        })
    return segments


def enforce_segment_lengths(
    candidate_idx: np.ndarray,
    boundary_scores: np.ndarray,
    min_steps: int,
    max_steps: int,
) -> np.ndarray:
    """Length-constrained boundary selection.

    1. Non-maximum suppression: visit candidates from strongest to weakest
       score and keep one only if it is at least `min_steps` from every kept
       boundary and from both ends of the video.
    2. Split: while some segment is longer than `max_steps`, add a boundary
       at its highest-scoring interior position (still `min_steps` from both
       of its ends).

    Indices follow the `boundary_scores` convention: t means a boundary
    between time-step t and t+1, so the next segment starts at t+1.
    """
    n = len(boundary_scores)  # = T - 1
    T = n + 1
    kept: List[int] = []
    for t in sorted((int(c) for c in candidate_idx), key=lambda c: -boundary_scores[c]):
        start = t + 1
        if start < min_steps or T - start < min_steps:
            continue
        if all(abs(start - (k + 1)) >= min_steps for k in kept):
            kept.append(t)

    changed = True
    while changed:
        changed = False
        cuts = [0] + sorted(k + 1 for k in kept) + [T]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= max_steps:
                continue
            lo, hi = a + min_steps - 1, b - min_steps - 1  # allowed boundary index range
            if hi < lo:
                continue
            t = lo + int(np.argmax(boundary_scores[lo:hi + 1]))
            kept.append(t)
            changed = True
            break
    return np.array(sorted(kept), dtype=np.int64)


def segments_from_scores(
    boundary_scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = None,
    min_distance: int = STAGE1.peak_min_distance_steps,
    adaptive_std: float = STAGE1.adaptive_threshold_std,
    min_segment_sec: float = STAGE1.min_segment_sec,
    max_segment_sec: float = STAGE1.max_segment_sec,
) -> List[Dict]:
    """Convenience wrapper: scores -> boundary indices -> segment list.

    `threshold=None` (default) uses the adaptive mean+std threshold to pick
    candidate peaks (see `boundary_scores_to_boundaries`); the candidates are
    then filtered by `enforce_segment_lengths`.
    """
    T = len(timestamps)
    if T == 0:
        return []
    step = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0
    min_steps = max(1, int(round(min_segment_sec / step)))
    max_steps = max(min_steps + 1, int(round(max_segment_sec / step)))

    candidates = boundary_scores_to_boundaries(boundary_scores, threshold, min_distance, adaptive_std)
    boundary_idx = enforce_segment_lengths(candidates, boundary_scores, min_steps, max_steps)
    return boundaries_to_segments(boundary_idx, timestamps, min_segment_len_steps=1)
