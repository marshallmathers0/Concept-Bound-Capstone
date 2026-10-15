"""
Evaluation metrics for Stage 1 (Pedagogical Boundary Detector) against
manually annotated ground-truth concept boundaries.

  - Boundary F1: precision/recall of predicted boundary points against
    ground-truth boundary points, with a tolerance window (in time-steps).
  - Segment IoU: for each ground-truth segment, the Intersection-over-Union
    with its best-matching predicted segment, averaged (mean IoU).
"""

from typing import Dict, List
import numpy as np

from config import EVAL


def boundary_f1(pred_boundary_idx: np.ndarray, gt_boundary_idx: np.ndarray,
                 tolerance: int = EVAL.boundary_tolerance_steps) -> Dict[str, float]:
    """Compute precision, recall and F1 for predicted vs. ground-truth
    boundary indices, matching within +/- `tolerance` time-steps (each
    ground-truth boundary can be matched at most once).
    """
    pred = sorted(int(p) for p in pred_boundary_idx)
    gt = sorted(int(g) for g in gt_boundary_idx)

    matched_gt = set()
    tp = 0
    for p in pred:
        for g in gt:
            if g in matched_gt:
                continue
            if abs(p - g) <= tolerance:
                matched_gt.add(g)
                tp += 1
                break

    fp = len(pred) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gt) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(pred) == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def segment_iou(seg_a: Dict, seg_b: Dict) -> float:
    """IoU between two segments given as {"start_time": ..., "end_time": ...}."""
    inter = max(0.0, min(seg_a["end_time"], seg_b["end_time"]) - max(seg_a["start_time"], seg_b["start_time"]))
    union = max(seg_a["end_time"], seg_b["end_time"]) - min(seg_a["start_time"], seg_b["start_time"])
    return inter / union if union > 0 else 0.0


def mean_segment_iou(pred_segments: List[Dict], gt_segments: List[Dict]) -> float:
    """For each ground-truth segment, find the predicted segment with the
    highest IoU and average these best-match IoUs across all ground-truth
    segments."""
    if not gt_segments:
        return 0.0
    if not pred_segments:
        return 0.0

    ious = []
    for gt_seg in gt_segments:
        best = max(segment_iou(gt_seg, pred_seg) for pred_seg in pred_segments)
        ious.append(best)
    return float(np.mean(ious))


def evaluate_boundaries(pred_segments: List[Dict], gt_segments: List[Dict],
                         total_steps: int, tolerance: int = EVAL.boundary_tolerance_steps) -> Dict[str, float]:
    """Convenience wrapper combining boundary F1 and mean segment IoU.

    `pred_segments` / `gt_segments` use the {start_idx, end_idx, start_time,
    end_time} schema produced by `boundaries_to_segments`. Internal
    boundaries are the `end_idx` of every segment except the last.
    """
    pred_boundaries = np.array([s["end_idx"] for s in pred_segments[:-1]], dtype=np.int64) if len(pred_segments) > 1 else np.array([], dtype=np.int64)
    gt_boundaries = np.array([s["end_idx"] for s in gt_segments[:-1]], dtype=np.int64) if len(gt_segments) > 1 else np.array([], dtype=np.int64)

    f1_metrics = boundary_f1(pred_boundaries, gt_boundaries, tolerance)
    miou = mean_segment_iou(pred_segments, gt_segments)

    return {**f1_metrics, "mean_iou": miou}


def _segment_ids(boundary_idx: np.ndarray, total_steps: int) -> np.ndarray:
    """(T,) segment id per time-step from boundary indices (boundary t lies
    between steps t and t+1)."""
    ids = np.zeros(total_steps, dtype=np.int64)
    for b in sorted(int(x) for x in boundary_idx):
        if 0 <= b < total_steps - 1:
            ids[b + 1:] += 1
    return ids


def pk_windowdiff(pred_boundary_idx: np.ndarray, gt_boundary_idx: np.ndarray,
                  total_steps: int, k: int = None) -> Dict[str, float]:
    """Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002),
    the standard text/topic segmentation error rates (lower is better).

    A window of `k` steps (default: half the mean gold segment length) slides
    over the sequence; Pk counts windows where prediction and gold disagree on
    whether the two ends are in the same segment, WindowDiff counts windows
    where they disagree on the number of boundaries inside.
    """
    gt = _segment_ids(gt_boundary_idx, total_steps)
    pred = _segment_ids(pred_boundary_idx, total_steps)
    if k is None:
        n_gt_segments = int(gt.max()) + 1
        k = max(2, int(round(total_steps / n_gt_segments / 2)))
    n = total_steps - k
    if n <= 0:
        return {"pk": 0.0, "windowdiff": 0.0, "k": k}
    pk_err = wd_err = 0
    for i in range(n):
        pk_err += (gt[i] == gt[i + k]) != (pred[i] == pred[i + k])
        wd_err += (gt[i + k] - gt[i]) != (pred[i + k] - pred[i])
    return {"pk": pk_err / n, "windowdiff": wd_err / n, "k": k}


def evaluate_boundaries_full(pred_segments: List[Dict], gt_segments: List[Dict],
                             total_steps: int, tolerances=(5, 10)) -> Dict[str, float]:
    """Boundary P/R/F1 at several tolerances (in steps), mean IoU, Pk and
    WindowDiff, plus segment counts - everything Stage 1 reports."""
    pred_b = np.array([s["end_idx"] for s in pred_segments[:-1]], dtype=np.int64) - 1
    gt_b = np.array([s["end_idx"] for s in gt_segments[:-1]], dtype=np.int64) - 1
    out = {}
    for tol in tolerances:
        m = boundary_f1(pred_b, gt_b, tol)
        out.update({f"precision@{tol}": m["precision"], f"recall@{tol}": m["recall"], f"f1@{tol}": m["f1"]})
    out["mean_iou"] = mean_segment_iou(pred_segments, gt_segments)
    out.update({k: v for k, v in pk_windowdiff(pred_b, gt_b, total_steps).items() if k != "k"})
    out["n_pred_segments"] = len(pred_segments)
    out["n_gt_segments"] = len(gt_segments)
    return out
