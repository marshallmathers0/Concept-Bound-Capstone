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
