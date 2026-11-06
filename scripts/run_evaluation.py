"""
Evaluate segmentation (Stage 1) and retrieval (Stage 2) on HUMAN GOLD data.

Only `data/annotations/gold/` is read as ground truth. Pseudo-labels and
pseudo-queries are never used here: they are derived from the system's own
inputs, so scoring against them measures nothing.

Stage 1: every segmenter vs the primary annotator's boundaries:
         P/R/F1 at +/-15s and +/-30s, mean IoU, Pk, WindowDiff.
Stage 2: every (segmenter x retriever) combination on the gold queries.
         The corpus is all evaluated videos together (course-level search).
         R@1/5/10 (IoU >= 0.3), R@1 at IoU >= 0.5, MRR, mIoU@1,
         with 95% bootstrap confidence intervals over queries.

Every run writes results/<run_name>/ with summary.json, a markdown table and
per-query rankings for error analysis.

Usage
-----
    python scripts/run_evaluation.py                          # test split
    python scripts/run_evaluation.py --split val              # tune on val, never on test
    python scripts/run_evaluation.py --dense-models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --rerank
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import EVAL, MODELS, PATHS, ensure_dirs
from src.evaluation.bench import (
    Segmenters, build_corpus, build_retrievers, evaluate_stage1, gold_queries,
)
from src.evaluation.retrieval_metrics import bootstrap_ci, paired_bootstrap_pvalue, per_query_metrics
from src.utils.gold import list_gold_videos, load_gold, load_splits

DEFAULT_SEGMENTERS = ["stage1", "pseudo", "topic_drift", "shot", "fixed_30", "fixed_60", "fixed_120", "gold"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fmt_ci(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--segmenters", nargs="+", default=DEFAULT_SEGMENTERS)
    parser.add_argument("--dense-models", nargs="+",
                        default=[MODELS.sentence_bert, MODELS.dense_baseline])
    parser.add_argument("--rerank", action="store_true", help="Add cross-encoder re-ranking of the hybrid run.")
    parser.add_argument("--stage1-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    parser.add_argument("--stage2-checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ensure_dirs()

    all_ids = [os.path.splitext(os.path.basename(p))[0]
               for p in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))]
    split = load_splits(all_ids)
    pool = all_ids if args.split == "all" else split[args.split]
    gold_ids = [v for v in pool if v in set(list_gold_videos())]

    if not gold_ids:
        print(f"No human gold annotations for the '{args.split}' split ({len(pool)} videos: {pool}).\n"
              f"Nothing can be evaluated honestly. Annotate them first:\n"
              f"  python scripts/make_annotation_sheets.py --videos {' '.join(pool)}\n"
              f"  (see docs/ANNOTATION_GUIDELINES.md), then python scripts/validate_annotations.py")
        sys.exit(2)
    missing = sorted(set(pool) - set(gold_ids))
    if missing:
        print(f"WARNING: {len(missing)} {args.split} video(s) have no gold yet and are excluded: {missing}")

    run = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.split}"
    out_dir = os.path.join(PATHS.results_dir, run)
    os.makedirs(out_dir, exist_ok=True)
    segs = Segmenters(args.stage1_checkpoint, args.device)
    seg_names = segs.names(args.segmenters)
    if "gold" in seg_names and not any((load_gold(v) or {}).get("boundaries_sec") for v in gold_ids):
        print("No gold boundaries yet: skipping the 'gold' (oracle) segmenter.")
        seg_names.remove("gold")

    # ---------------- Stage 1 ----------------
    stage1 = evaluate_stage1(gold_ids, seg_names, segs)
    lines = [f"# Evaluation `{run}` (split={args.split}, videos={len(gold_ids)}, commit={git_commit()})", "",
             "## Stage 1: segmentation vs gold boundaries", "",
             "| segmenter | F1@15s | F1@30s | Pk (lower=better) | WindowDiff | mean IoU | #segments (gold) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in stage1["summary"].items():
        lines.append(f"| {name} | {m['f1@5']:.3f} | {m['f1@10']:.3f} | {m['pk']:.3f} | {m['windowdiff']:.3f} | "
                     f"{m['mean_iou']:.3f} | {m['n_pred_segments']:.1f} ({m['n_gt_segments']:.1f}) |")

    # ---------------- Stage 2 ----------------
    queries = gold_queries(gold_ids)
    stage2, per_query_rows = {}, []
    lines += ["", f"## Stage 2: retrieval ({len(queries)} gold queries, corpus = {len(gold_ids)} videos)", "",
              "| segmenter | retriever | R@1 [95% CI] | R@5 [95% CI] | R@10 | R@1 (IoU≥0.5) | MRR [95% CI] | mIoU@1 | #units |",
              "|---|---|---|---|---|---|---|---|---|"]
    if not queries:
        lines.append("| (no gold queries) |||||||||")
    for seg_name in seg_names if queries else []:
        units = build_corpus(gold_ids, seg_name, segs)
        retrievers = build_retrievers(units, args.dense_models, args.stage2_checkpoint, args.rerank, args.device)
        for r in retrievers:
            per_q = []
            for q in queries:
                res = r.search(q["query"], top_k=max(EVAL.retrieval_ks))
                m = per_query_metrics(res, q)
                per_q.append(m)
                per_query_rows.append({"segmenter": seg_name, "retriever": r.name, **q, **m,
                                       "top5": [(x["video_id"], x["start_time"], x["end_time"]) for x in res[:5]]})
            key = f"{seg_name}|{r.name}"
            stage2[key] = {"segmenter": seg_name, "retriever": r.name, "n_units": len(units),
                           "per_query": per_q}
            ci = {k: bootstrap_ci([p[k] for p in per_q]) for k in per_q[0]}
            stage2[key]["metrics"] = {k: {"mean": v[0], "ci95": [v[1], v[2]]} for k, v in ci.items()}
            lines.append(f"| {seg_name} | {r.name} | {fmt_ci(*ci['R@1'])} | {fmt_ci(*ci['R@5'])} | "
                         f"{ci['R@10'][0]:.3f} | {ci['R@1_iou0.5'][0]:.3f} | {fmt_ci(*ci['MRR'])} | "
                         f"{ci['mIoU@1'][0]:.3f} | {len(units)} |")

    # Is the trained system significantly better than plain BM25 on fixed windows?
    ref = "fixed_60|bm25"
    if ref in stage2:
        lines += ["", f"### Paired bootstrap vs `{ref}` (MRR, one-sided p that the system is not better)", ""]
        for key, v in stage2.items():
            if key == ref:
                continue
            p = paired_bootstrap_pvalue([x["MRR"] for x in v["per_query"]],
                                        [x["MRR"] for x in stage2[ref]["per_query"]])
            lines.append(f"- `{key}`: p = {p:.3f}")

    # Breakdown by query type for the best run.
    if stage2:
        best = max(stage2, key=lambda k: stage2[k]["metrics"]["MRR"]["mean"])
        lines += ["", f"### Query-type breakdown for best run `{best}` (by MRR)", "", "| type | n | R@1 | MRR |", "|---|---|---|---|"]
        rows = [r for r in per_query_rows if f"{r['segmenter']}|{r['retriever']}" == best]
        for t in sorted(set(r["type"] for r in rows)):
            sel = [r for r in rows if r["type"] == t]
            lines.append(f"| {t} | {len(sel)} | {np.mean([r['R@1'] for r in sel]):.3f} | {np.mean([r['MRR'] for r in sel]):.3f} |")

    lines += ["", "Hit = top result in the right video with temporal IoU ≥ "
              f"{EVAL.retrieval_iou_threshold} against the gold moment. CIs: percentile bootstrap over queries."]

    summary = {
        "run": run, "split": args.split, "videos": gold_ids, "excluded_no_gold": missing,
        "commit": git_commit(), "args": vars(args), "eval_config": EVAL.__dict__,
        "stage1": stage1,
        "stage2": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in stage2.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "per_query.jsonl"), "w", encoding="utf-8") as f:
        for row in per_query_rows:
            f.write(json.dumps(row, default=float) + "\n")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
