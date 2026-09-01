"""
End-to-end evaluation: compares the trained Pedagogical Boundary Detector
and Cross-Modal Retrieval model against the baselines described in the
proposal's Section 5.2.

Stage 1 (segmentation):
    - Pedagogical Boundary Detector (trained)
    - Fixed-window segmentation (60s)
    - Shot-boundary detection (visual-signal fallback)
  Metrics: boundary F1, mean segment IoU (vs. manually annotated boundaries
  in `data/annotations/<video_id>_boundaries.json`)

Stage 2 (retrieval):
    - Cross-Modal Segment Retrieval over Stage-1 segments (trained)
    - Sliding-window retrieval without concept-aware segmentation
  Metrics: Recall@1/5/10, mean IoU@1 (vs. (query, segment) pairs in
  `data/annotations/<video_id>_queries.json`)

Usage
-----
    python scripts/run_evaluation.py
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import EVAL, PATHS, STAGE1, ensure_dirs
from src.utils.io_utils import load_features, load_json

from src.stage1_boundary.inference import load_stage1_model, segment_video
from src.stage1_boundary.pseudo_labels import boundaries_from_annotations
from src.stage1_boundary.segment import boundaries_to_segments
from src.baselines.fixed_window import fixed_window_segments
from src.baselines.shot_detection import shot_segments_from_visual_signal
from src.evaluation.boundary_metrics import evaluate_boundaries

from src.stage2_retrieval.retrieval import LectureRetriever
from src.baselines.sliding_window_retrieval import build_sliding_window_index, search_sliding_window
from src.evaluation.retrieval_metrics import evaluate_retrieval


def list_feature_video_ids(features_dir):
    paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def evaluate_stage1(video_ids, args):
    if not os.path.exists(args.stage1_checkpoint):
        print(f"[Stage 1] checkpoint not found at {args.stage1_checkpoint}, skipping.")
        return

    model = load_stage1_model(args.stage1_checkpoint, args.device)

    results = {"pedagogical": [], "fixed_window": [], "shot_detection": []}

    for video_id in video_ids:
        ann_path = os.path.join(PATHS.annotations_dir, f"{video_id}_boundaries.json")
        if not os.path.exists(ann_path):
            continue

        feats = load_features(os.path.join(args.features_dir, f"{video_id}.npz"))
        timestamps = feats["timestamps"]
        visual_signal = feats["visual_signal"]

        ann = load_json(ann_path)
        gt_is_boundary = boundaries_from_annotations(timestamps, ann.get("boundary_times_sec", []))
        gt_boundary_idx = np.where(gt_is_boundary == 1)[0]
        gt_segments = boundaries_to_segments(gt_boundary_idx, timestamps, min_segment_len_steps=1)

        pred_segments = segment_video(video_id, model, args.features_dir, args.device,
                                        threshold=args.boundary_threshold)
        fixed_segments = fixed_window_segments(timestamps, window_sec=args.fixed_window_sec)
        shot_segments = shot_segments_from_visual_signal(timestamps, visual_signal)

        for key, segs in [("pedagogical", pred_segments),
                           ("fixed_window", fixed_segments),
                           ("shot_detection", shot_segments)]:
            metrics = evaluate_boundaries(segs, gt_segments, total_steps=len(timestamps),
                                           tolerance=EVAL.boundary_tolerance_steps)
            results[key].append(metrics)

    print("\n=== Stage 1: Boundary Detection ===")
    for key, all_metrics in results.items():
        if not all_metrics:
            continue
        avg = {m: float(np.mean([r[m] for r in all_metrics])) for m in ["precision", "recall", "f1", "mean_iou"]}
        print(f"  {key:16s}: F1={avg['f1']:.3f}  Precision={avg['precision']:.3f}  "
              f"Recall={avg['recall']:.3f}  mean_IoU={avg['mean_iou']:.3f}  (n={len(all_metrics)} videos)")


def evaluate_stage2(video_ids, args):
    if not os.path.exists(args.stage2_checkpoint):
        print(f"\n[Stage 2] checkpoint not found at {args.stage2_checkpoint}, skipping.")
        return
    if not os.path.exists(os.path.join(args.index_dir, "segments.index")):
        print(f"\n[Stage 2] FAISS index not found in {args.index_dir}, skipping. Run build_index.py first.")
        return

    retriever = LectureRetriever(args.index_dir, args.stage2_checkpoint, args.device)

    sw_video_ids = [v for v in video_ids if os.path.exists(os.path.join(args.features_dir, f"{v}.npz"))]
    sw_embeddings, sw_metadata = build_sliding_window_index(sw_video_ids, features_dir=args.features_dir, device=args.device)

    retrieved_main, retrieved_sw, ground_truth = [], [], []

    for video_id in video_ids:
        ann_path = os.path.join(PATHS.annotations_dir, f"{video_id}_queries.json")
        if not os.path.exists(ann_path):
            continue
        ann = load_json(ann_path)

        for pair in ann["pairs"]:
            results_main = retriever.search(pair["query"], top_k=max(EVAL.retrieval_ks))
            results_sw = search_sliding_window(pair["query"], sw_embeddings, sw_metadata,
                                                 top_k=max(EVAL.retrieval_ks), device=args.device) if len(sw_embeddings) else []

            retrieved_main.append(results_main)
            retrieved_sw.append(results_sw)
            ground_truth.append({"video_id": video_id, "start_time": pair["start_time"], "end_time": pair["end_time"]})

    print("\n=== Stage 2: Cross-Modal Retrieval ===")
    if retrieved_main:
        main_metrics = evaluate_retrieval(retrieved_main, ground_truth)
        print(f"  cross_modal_fusion : " + "  ".join(f"{k}={v:.3f}" for k, v in main_metrics.items()) +
              f"  (n={len(ground_truth)} queries)")
    if retrieved_sw:
        sw_metrics = evaluate_retrieval(retrieved_sw, ground_truth)
        print(f"  sliding_window     : " + "  ".join(f"{k}={v:.3f}" for k, v in sw_metrics.items()))


def evaluate_gold_standard(args):
    """Evaluate retrieval on human-annotated gold-standard annotations.

    Gold annotations are loaded from data/annotations/gold_standard/<video_id>_gold.json
    and evaluated with ±tolerance_sec (default ±10s).
    """
    gold_dir = os.path.join(PATHS.annotations_dir, "gold_standard")
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*_gold.json")))

    if not gold_files:
        print(f"\n[Gold Standard] No annotation files found in {gold_dir}")
        print(f"  Create annotations following the template in {gold_dir}/README.md")
        return

    if not os.path.exists(os.path.join(args.index_dir, "segments.index")):
        print(f"\n[Gold Standard] FAISS index not found in {args.index_dir}, skipping.")
        return

    retriever = LectureRetriever(args.index_dir, args.stage2_checkpoint, args.device)

    retrieved_all, ground_truth_all = [], []
    source_counts = {"nptel": 0, "local": 0, "other": 0}

    for gold_path in gold_files:
        ann = load_json(gold_path)
        video_id = ann.get("video_id", os.path.basename(gold_path).replace("_gold.json", ""))
        source = ann.get("source", "other")

        # Filter by split if specified
        if args.split and args.split != source:
            continue

        source_counts[source] = source_counts.get(source, 0) + len(ann.get("annotations", []))

        for item in ann.get("annotations", []):
            results = retriever.search(item["query"], top_k=max(EVAL.retrieval_ks))
            retrieved_all.append(results)
            ground_truth_all.append({
                "video_id": video_id,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
            })

    if not retrieved_all:
        print(f"\n[Gold Standard] No queries found (split={args.split or 'all'})")
        return

    tolerance = EVAL.retrieval_tolerance_sec
    metrics = evaluate_retrieval(retrieved_all, ground_truth_all,
                                  tolerance_sec=tolerance)

    from src.evaluation.retrieval_metrics import precision_recall_f1
    prf = precision_recall_f1(retrieved_all, ground_truth_all,
                               tolerance_sec=tolerance)

    print(f"\n=== Gold-Standard Evaluation (±{tolerance:.0f}s tolerance) ===")
    split_label = f" [{args.split}]" if args.split else ""
    print(f"  Queries: {len(ground_truth_all)}{split_label}")
    print(f"  Sources: " + ", ".join(f"{k}={v}" for k, v in source_counts.items() if v > 0))
    print(f"  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(f"  Precision={prf['precision']:.3f}  Recall={prf['recall']:.3f}  "
          f"F1={prf['f1']:.3f}  (TP={prf['tp']} FP={prf['fp']} FN={prf['fn']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=PATHS.features_dir)
    parser.add_argument("--index-dir", default=PATHS.index_dir)
    parser.add_argument("--stage1-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    parser.add_argument("--stage2-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    parser.add_argument("--boundary-threshold", type=float, default=None,
                         help="Fixed boundary-score threshold. Default: adaptive (mean + std).")
    parser.add_argument("--fixed-window-sec", type=float, default=60.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gold-standard", action="store_true",
                         help="Evaluate on gold-standard human annotations with ±10s tolerance.")
    parser.add_argument("--split", choices=["nptel", "local"], default=None,
                         help="Evaluate only on a specific data source split (nptel=train, local=test).")
    args = parser.parse_args()

    ensure_dirs()
    video_ids = list_feature_video_ids(args.features_dir)
    if not video_ids:
        print(f"No feature files found in {args.features_dir}.")
        return

    evaluate_stage1(video_ids, args)
    evaluate_stage2(video_ids, args)

    if args.gold_standard:
        evaluate_gold_standard(args)


if __name__ == "__main__":
    main()

