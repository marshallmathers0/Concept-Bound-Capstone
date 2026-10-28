"""
Collect every number the paper reports into results/paper_results.json.

paper/build_paper.js reads that file and fills the tables and text; anything
missing stays a yellow placeholder. Re-run after every new annotation batch:

    python scripts/run_evaluation.py --split test --rerank
    python scripts/run_ablations.py --split test
    python scripts/measure_runtime.py
    python scripts/collect_paper_results.py
    node paper/build_paper.js && node paper/build_paper.js --blind
"""

import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import PATHS
from src.utils.gold import list_gold_videos, load_gold
from src.utils.io_utils import load_features
from src.utils.segment_text import clean_asr, has_raw_asr

HYBRID = "hybrid(bm25+dense:all-MiniLM-L6-v2)"
TABLE_RETRIEVAL = [  # (row label, "segmenter|retriever" key in run_evaluation's summary)
    ("Fixed 60 s + BM25", "fixed_60|bm25"),
    ("Fixed 60 s + BGE", "fixed_60|dense:bge-small-en-v1.5"),
    ("Fixed 60 s + RRF + rerank", f"fixed_60|{HYBRID}+rerank"),
    ("ConceptBound seg. + BM25", "stage1|bm25"),
    ("ConceptBound seg. + RRF + rerank", f"stage1|{HYBRID}+rerank"),
    ("ConceptBound full (ours)", "stage1|fusion"),
    ("Ground-truth seg. + RRF + rerank (oracle)", f"gold|{HYBRID}+rerank"),
]
TABLE_SEG = [("Fixed 60 s", "fixed_60"), ("Fixed 120 s", "fixed_120"), ("Shot change", "shot"),
             ("Topic drift [7]", "topic_drift"), ("Pseudo-label peaks", "pseudo"),
             ("ConceptBound Stage 1 (ours)", "stage1")]


def latest_eval(split):
    runs = sorted(glob.glob(os.path.join(PATHS.results_dir, f"*_{split}", "summary.json")) +
                  glob.glob(os.path.join(PATHS.results_dir, f"paper_{split}", "summary.json")), key=os.path.getmtime)
    return json.load(open(runs[-1])) if runs else None


def corpus_stats():
    rows = []
    for path in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz"))):
        f = load_features(path)
        if not has_raw_asr(f):
            continue
        prov = json.loads(str(f["provenance"]))
        segs = clean_asr(f)
        rows.append({"words": sum(len(s.text.split()) for s in segs),
                     "removed": len(f["asr_segments"]) - len(segs),
                     "ocr_cov": prov.get("ocr_coverage", 0.0),
                     "slides": len(set(str(x) for x in f["ocr_texts"] if str(x).strip()))})
    n_all = len(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))
    if not rows:
        return None
    col = lambda k: [r[k] for r in rows]
    return {"n_processed": len(rows), "n_lectures": n_all, "hours": round(n_all * 64 / 60, 1),
            "words_min": min(col("words")), "words_max": max(col("words")), "words_mean": float(np.mean(col("words"))),
            "removed_min": min(col("removed")), "removed_max": max(col("removed")),
            "ocr_min": min(col("ocr_cov")), "ocr_max": max(col("ocr_cov")),
            "slides_min": min(col("slides")), "slides_max": max(col("slides"))}


def gold_stats():
    vids = list_gold_videos()
    qs, nb, ann2 = [], [], 0
    for v in vids:
        g = load_gold(v)
        qs += [q["end_time"] - q["start_time"] for q in g["queries"]]
        nb.append(len(g["boundaries_sec"]))
        ann2 += len(g["annotators"]) > 1
    return {"lectures": len(vids), "queries": len(qs), "median_span": float(np.median(qs)) if qs else None,
            "mean_span": float(np.mean(qs)) if qs else None, "with_boundaries": sum(1 for b in nb if b),
            "double_annotated": ann2}


def main():
    out = {"corpus": corpus_stats(), "gold": gold_stats()}
    ev = latest_eval("test")
    if ev:
        s2 = ev["stage2"]
        out["eval_run"] = ev["run"]
        out["retrieval"] = {}
        for label, key in TABLE_RETRIEVAL:
            if key in s2:
                m = s2[key]["metrics"]
                out["retrieval"][label] = {k: m[k]["mean"] for k in ("R@1", "R@5", "MRR", "mIoU@1")}
                out["retrieval"][label]["ci_R@1"] = m["R@1"]["ci95"]
        out["segmentation"] = {label: ev["stage1"]["summary"][key]
                               for label, key in TABLE_SEG if key in ev["stage1"]["summary"]}
    abl = os.path.join(PATHS.results_dir, "ablations_test", "ablations.json")
    if os.path.exists(abl):
        out["ablations"] = json.load(open(abl))
    for name in ("runtime", "asr_wer"):
        p = os.path.join(PATHS.results_dir, f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    path = os.path.join(PATHS.results_dir, "paper_results.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(json.dumps({k: (v if k in ("corpus", "gold") else "present") for k, v in out.items()}, indent=2, default=float))
    print("Saved", path)


if __name__ == "__main__":
    main()
