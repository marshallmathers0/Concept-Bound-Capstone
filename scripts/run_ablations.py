"""
Run every ablation in the paper and write one results table.

Ablations (all evaluated on gold data of the chosen split, multi-video search):
  A. Stage 1 signal removal  - retrain Stage 1 without OCR / drift / visual / CLIP
  B. Stage 1 encoder         - BiLSTM vs CNN vs Transformer
  C. Stage 2 fusion design   - 3-token attention vs concat vs pairwise (Arora et al.)
  D. Segment length bounds   - post-processing limits, no retraining
  E. Length-matched control  - fixed windows with the same mean length as Stage 1
  F. Seeds                   - main Stage 1 + Stage 2 with 3 seeds (mean +/- std)

Segmentation variants are scored with the RRF text retriever (BM25 + MiniLM),
so differences come from the segmentation; fusion variants use their own
Stage-2 model on the main Stage-1 segments. Boundary metrics are added for
every Stage-1 variant whenever the split has gold boundaries.

Checkpoints go to checkpoints/ablations/, results to
results/ablations_<split>/ablations.json and .md.

Usage
-----
    python scripts/run_ablations.py --split test
    python scripts/run_ablations.py --split test --skip-training   # reuse checkpoints
    python scripts/run_ablations.py --only C E                       # subset
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import torch

from config import EVAL, MODELS, PATHS, STAGE1, ensure_dirs
from src.evaluation.bench import (
    BM25Retriever, DenseRetriever, FusionRetriever, HybridRetriever, Segmenters,
    build_corpus, evaluate_stage1, gold_queries,
)
from src.evaluation.retrieval_metrics import bootstrap_ci, per_query_metrics
from src.utils.gold import list_gold_videos, load_splits

ABL_CKPT = os.path.join(PATHS.checkpoints_dir, "ablations")


def run(cmd):
    print("  $ python " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout


def train_stage1(name, extra, epochs, skip):
    ckpt = os.path.join(ABL_CKPT, f"stage1_{name}.pt")
    if not (skip and os.path.exists(ckpt)):
        run(["-m", "src.stage1_boundary.train", "--epochs", str(epochs), "--checkpoint", ckpt] + extra)
    return ckpt


def train_stage2(name, extra, epochs, skip):
    ckpt = os.path.join(ABL_CKPT, f"stage2_{name}.pt")
    if not (skip and os.path.exists(ckpt)):
        run(["-m", "src.stage2_retrieval.train", "--epochs", str(epochs), "--checkpoint", ckpt] + extra)
    return ckpt


def score(retriever, queries):
    per_q = [per_query_metrics(retriever.search(q["query"], top_k=max(EVAL.retrieval_ks)), q) for q in queries]
    out = {}
    for k in ("R@1", "R@5", "R@10", "MRR", "mIoU@1"):
        m, lo, hi = bootstrap_ci([p[k] for p in per_q])
        out[k] = {"mean": m, "ci95": [lo, hi]}
    return out, per_q


def text_retriever(units, device):
    bm25 = BM25Retriever(units)
    dense = DenseRetriever(units, MODELS.sentence_bert, device)
    return HybridRetriever(bm25, dense)


def seg_row(label, segs, seg_name, video_ids, queries, device, rows, group):
    units = build_corpus(video_ids, seg_name, segs)
    metrics, _ = score(text_retriever(units, device), queries)
    mean_len = float(np.mean([u["end_time"] - u["start_time"] for u in units]))
    row = {"group": group, "variant": label, "retriever": "RRF(BM25+MiniLM)", "n_units": len(units),
           "mean_unit_sec": mean_len, **{k: v["mean"] for k, v in metrics.items()},
           "ci95": {k: v["ci95"] for k, v in metrics.items()}}
    st1 = evaluate_stage1(video_ids, [seg_name], segs)["summary"].get(seg_name)
    if st1:
        row.update({"f1@15s": st1["f1@5"], "pk": st1["pk"], "windowdiff": st1["windowdiff"]})
    rows.append(row)
    print(f"  [{group}] {label}: R@1={row['R@1']:.3f} MRR={row['MRR']:.3f} units={len(units)} mean={mean_len:.0f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--only", nargs="*", default=list("ABCDEF"))
    ap.add_argument("--stage1-epochs", type=int, default=30)
    ap.add_argument("--stage2-epochs", type=int, default=20)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ensure_dirs()
    os.makedirs(ABL_CKPT, exist_ok=True)

    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))]
    split = load_splits(all_ids)
    pool = all_ids if args.split == "all" else split[args.split]
    vids = [v for v in pool if v in set(list_gold_videos())]
    queries = gold_queries(vids)
    if not queries:
        sys.exit(f"No gold queries in split '{args.split}'.")
    print(f"Ablations on {len(vids)} lectures, {len(queries)} gold queries: {vids}")
    rows, t0 = [], time.time()
    main_s1 = os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt")
    main_s2 = os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")

    if "A" in args.only:
        seg_row("full (all signals)", Segmenters(main_s1, args.device), "stage1", vids, queries, args.device, rows, "A: signal removal")
        for sig in ("ocr", "drift", "visual", "clip"):
            ck = train_stage1(f"drop_{sig}", ["--drop-signals", sig], args.stage1_epochs, args.skip_training)
            seg_row(f"without {sig}", Segmenters(ck, args.device), "stage1", vids, queries, args.device, rows, "A: signal removal")

    if "B" in args.only:
        for enc in ("bilstm", "cnn", "transformer"):
            ck = train_stage1(f"enc_{enc}", ["--encoder-type", enc], args.stage1_epochs, args.skip_training)
            seg_row(enc, Segmenters(ck, args.device), "stage1", vids, queries, args.device, rows, "B: Stage-1 encoder")

    if "C" in args.only:
        segs = Segmenters(main_s1, args.device)
        units = build_corpus(vids, "stage1", segs)
        for mode in ("cross_attention", "concat", "pairwise"):
            ck = train_stage2(f"fusion_{mode}", ["--fusion-mode", mode], args.stage2_epochs, args.skip_training)
            metrics, _ = score(FusionRetriever(units, ck, args.device), queries)
            rows.append({"group": "C: fusion design", "variant": mode, "retriever": "Stage-2 fusion", "n_units": len(units),
                         **{k: v["mean"] for k, v in metrics.items()}, "ci95": {k: v["ci95"] for k, v in metrics.items()}})
            print(f"  [C] {mode}: R@1={metrics['R@1']['mean']:.3f} MRR={metrics['MRR']['mean']:.3f}", flush=True)

    if "D" in args.only:
        for lo, hi in ((30, 300), (60, 600), (120, 900)):
            seg_row(f"{lo}-{hi} s", Segmenters(main_s1, args.device, lo, hi), "stage1", vids, queries, args.device, rows, "D: segment length")

    if "E" in args.only:
        ref = seg_row("Stage 1", Segmenters(main_s1, args.device), "stage1", vids, queries, args.device, rows, "E: length-matched")
        win = round(ref["mean_unit_sec"])
        seg_row(f"fixed {win} s (matched)", Segmenters(None, args.device), f"fixed_{win}", vids, queries, args.device, rows, "E: length-matched")

    if "F" in args.only:
        per_seed = []
        for seed in args.seeds:
            c1 = train_stage1(f"seed{seed}", ["--seed", str(seed)], args.stage1_epochs, args.skip_training)
            segs = Segmenters(c1, args.device)
            units = build_corpus(vids, "stage1", segs)
            text_m, _ = score(text_retriever(units, args.device), queries)
            c2 = train_stage2(f"seed{seed}", ["--seed", str(seed)], args.stage2_epochs, args.skip_training)
            fus_m, _ = score(FusionRetriever(units, c2, args.device), queries)
            per_seed.append({"seed": seed, "text_R@1": text_m["R@1"]["mean"], "text_MRR": text_m["MRR"]["mean"],
                             "fusion_R@1": fus_m["R@1"]["mean"], "fusion_MRR": fus_m["MRR"]["mean"]})
            print(f"  [F] seed {seed}: {per_seed[-1]}", flush=True)
        for key in ("text_R@1", "text_MRR", "fusion_R@1", "fusion_MRR"):
            vals = [p[key] for p in per_seed]
            rows.append({"group": "F: seeds", "variant": key, "mean": float(np.mean(vals)),
                         "std": float(np.std(vals)), "per_seed": vals})

    out_dir = os.path.join(PATHS.results_dir, f"ablations_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"split": args.split, "videos": vids, "n_queries": len(queries), "runtime_sec": time.time() - t0,
            "stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs}
    json.dump({"meta": meta, "rows": rows}, open(os.path.join(out_dir, "ablations.json"), "w"), indent=2)

    lines = [f"# Ablations (split={args.split}, {len(vids)} lectures, {len(queries)} queries)", "",
             "| group | variant | R@1 | MRR | mIoU@1 | F1@15s | Pk | units | mean unit (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"].startswith("F"):
            continue
        f = lambda k: f"{r[k]:.3f}" if k in r else "–"
        lines.append(f"| {r['group']} | {r['variant']} | {f('R@1')} | {f('MRR')} | {f('mIoU@1')} | {f('f1@15s')} | "
                     f"{f('pk')} | {r.get('n_units', '')} | {r.get('mean_unit_sec', 0):.0f} |")
    seeds = [r for r in rows if r["group"].startswith("F")]
    if seeds:
        lines += ["", "| seeds | mean | std |", "|---|---|---|"] + [f"| {r['variant']} | {r['mean']:.3f} | {r['std']:.3f} |" for r in seeds]
    open(os.path.join(out_dir, "ablations.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_dir} ({meta['runtime_sec'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
