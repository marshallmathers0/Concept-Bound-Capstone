"""
Evaluation metrics for Stage 2 (Cross-Modal Segment Retrieval) against
manually annotated (query, ground-truth segment) pairs.

  - Recall@k: fraction of queries for which at least one of the top-k
    retrieved segments overlaps the ground-truth segment with IoU >=
    `iou_threshold`.
  - Mean IoU: average IoU between the top-1 retrieved segment and the
    ground-truth segment, across all queries.
  - Precision / Recall / F1: segment-level metrics with a configurable
    ±tolerance_sec window for gold-standard evaluation.
"""

from typing import Dict, List
import numpy as np

from config import EVAL


def temporal_iou(a_start, a_end, b_start, b_end, tolerance_sec: float = 0.0) -> float:
    """Compute temporal IoU, optionally expanding the ground-truth window.

    When tolerance_sec > 0, the ground-truth interval [b_start, b_end] is
    expanded to [b_start - tolerance_sec, b_end + tolerance_sec] before
    computing IoU. This implements the ±10s tolerance for gold-standard
    evaluation.
    """
    b_start_expanded = b_start - tolerance_sec
    b_end_expanded = b_end + tolerance_sec
    inter = max(0.0, min(a_end, b_end_expanded) - max(a_start, b_start_expanded))
    union = max(a_end, b_end_expanded) - min(a_start, b_start_expanded)
    return inter / union if union > 0 else 0.0


def precision_recall_f1(retrieved: List[List[Dict]], ground_truth: List[Dict],
                         tolerance_sec: float = EVAL.retrieval_tolerance_sec,
                         iou_threshold: float = EVAL.retrieval_iou_threshold) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 for retrieval with ±tolerance_sec window.

    A retrieved segment is a 'hit' if its temporal IoU with the ground-truth
    segment (expanded by ±tolerance_sec) exceeds iou_threshold.

    This is the primary evaluation metric for the gold-standard test set,
    designed for human-annotated ground-truth with a ±10-second tolerance.
    """
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp, fp, fn = 0, 0, 0
    for results, gt in zip(retrieved, ground_truth):
        hit = False
        for r in results[:1]:  # top-1 result
            if r["video_id"] != gt["video_id"]:
                continue
            iou = temporal_iou(r["start_time"], r["end_time"],
                               gt["start_time"], gt["end_time"],
                               tolerance_sec=tolerance_sec)
            if iou >= iou_threshold:
                hit = True
                break
        if hit:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def recall_at_k(retrieved: List[List[Dict]], ground_truth: List[Dict],
                 k: int, iou_threshold: float = EVAL.retrieval_iou_threshold) -> float:
    """
    retrieved: for each query, a ranked list of result dicts with
               video_id/start_time/end_time (already truncated/sliced to top-k
               by the caller, or full list - this function takes [:k]).
    ground_truth: for each query, a dict with video_id/start_time/end_time.
    """
    if not retrieved:
        return 0.0

    hits = 0
    for results, gt in zip(retrieved, ground_truth):
        hit = False
        for r in results[:k]:
            if r["video_id"] != gt["video_id"]:
                continue
            iou = temporal_iou(r["start_time"], r["end_time"], gt["start_time"], gt["end_time"])
            if iou >= iou_threshold:
                hit = True
                break
        hits += int(hit)
    return hits / len(retrieved)


def mean_iou_at_1(retrieved: List[List[Dict]], ground_truth: List[Dict]) -> float:
    if not retrieved:
        return 0.0
    ious = []
    for results, gt in zip(retrieved, ground_truth):
        if not results:
            ious.append(0.0)
            continue
        top = results[0]
        if top["video_id"] != gt["video_id"]:
            ious.append(0.0)
            continue
        ious.append(temporal_iou(top["start_time"], top["end_time"], gt["start_time"], gt["end_time"]))
    return float(np.mean(ious))


def evaluate_retrieval(retrieved: List[List[Dict]], ground_truth: List[Dict],
                        ks: List[int] = None, iou_threshold: float = EVAL.retrieval_iou_threshold,
                        tolerance_sec: float = 0.0) -> Dict[str, float]:
    ks = ks or EVAL.retrieval_ks
    metrics = {f"recall@{k}": recall_at_k(retrieved, ground_truth, k, iou_threshold) for k in ks}
    metrics["mean_iou@1"] = mean_iou_at_1(retrieved, ground_truth)

    # Add P/R/F1 with tolerance for gold-standard evaluation
    if tolerance_sec > 0:
        prf = precision_recall_f1(retrieved, ground_truth, tolerance_sec, iou_threshold)
        metrics.update({f"gold_{k}": v for k, v in prf.items() if isinstance(v, float)})

    return metrics


def first_hit_rank(results: List[Dict], gt: Dict, iou_threshold: float = EVAL.retrieval_iou_threshold):
    """1-based rank of the first result overlapping the gold moment with
    IoU >= threshold, or None."""
    for rank, r in enumerate(results, start=1):
        if r["video_id"] == gt["video_id"] and \
                temporal_iou(r["start_time"], r["end_time"], gt["start_time"], gt["end_time"]) >= iou_threshold:
            return rank
    return None


def per_query_metrics(results: List[Dict], gt: Dict, ks=None) -> Dict[str, float]:
    """Metrics for one query; average these over queries (and bootstrap them)."""
    ks = ks or EVAL.retrieval_ks
    rank = first_hit_rank(results, gt, EVAL.retrieval_iou_threshold)
    rank50 = first_hit_rank(results, gt, 0.5)
    top = results[0] if results else None
    iou1 = (temporal_iou(top["start_time"], top["end_time"], gt["start_time"], gt["end_time"])
            if top is not None and top["video_id"] == gt["video_id"] else 0.0)
    m = {f"R@{k}": float(rank is not None and rank <= k) for k in ks}
    m["R@1_iou0.5"] = float(rank50 == 1)
    m["MRR"] = 1.0 / rank if rank else 0.0
    m["mIoU@1"] = iou1
    m["video_hit@1"] = float(top is not None and top["video_id"] == gt["video_id"])
    return m


def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap (mean, lo, hi) over per-query values."""
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_pvalue(a: List[float], b: List[float], n_boot: int = 5000, seed: int = 0) -> float:
    """One-sided p-value that system A is not better than B on the same queries
    (paired bootstrap over queries)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((diffs <= 0).mean())
