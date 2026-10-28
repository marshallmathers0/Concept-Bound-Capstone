"""
Measure hardware, per-stage processing time and query latency for the paper.

  - hardware: CPU model, cores, RAM, GPU (if any)
  - feature extraction: OCR / ASR seconds per lecture, parsed from the
    refresh_features logs in logs/
  - Stage 1 inference and segment-feature time for one lecture
  - query latency (median / p95) of each retriever over the gold queries

Writes results/runtime.json.

Usage
-----
    python scripts/measure_runtime.py
"""

import glob
import json
import os
import platform
import re
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import MODELS, PATHS, ensure_dirs
from src.evaluation.bench import (
    BM25Retriever, DenseRetriever, FusionRetriever, HybridRetriever, Segmenters, build_corpus, gold_queries,
)
from src.utils.gold import list_gold_videos


def hardware():
    cpu = platform.processor()
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_Processor).Name; "
                                       "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"],
                                      text=True, timeout=60).strip().splitlines()
        cpu, ram = out[0].strip(), f"{out[-1].strip()} GB"
    except Exception:
        ram = "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    return {"cpu": cpu, "cores": os.cpu_count(), "ram": ram, "gpu": gpu, "torch": torch.__version__}


def extraction_times():
    ocr, asr = [], []
    for log in glob.glob(os.path.join(os.path.dirname(PATHS.results_dir), "logs", "refresh_features*.log")):
        text = open(log, encoding="utf-8", errors="ignore").read()
        ocr += [float(x) for x in re.findall(r"OCR refreshed in (\d+)s", text)]
        asr += [float(x) for x in re.findall(r"ASR refreshed in (\d+)s", text)]
    stat = lambda v: {"n": len(v), "median_sec": float(np.median(v)), "min_sec": min(v), "max_sec": max(v)} if v else None
    return {"ocr": stat(ocr), "asr": stat(asr)}


def main():
    ensure_dirs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"hardware": hardware(), "extraction": extraction_times()}
    vids = list_gold_videos()
    queries = gold_queries(vids)

    s1 = os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt")
    segs = Segmenters(s1, dev)
    t = time.perf_counter()
    units = build_corpus(vids[:1], "stage1", segs)
    out["stage1_segment_one_lecture_sec"] = time.perf_counter() - t

    units = build_corpus(vids, "stage1", segs)
    bm25 = BM25Retriever(units)
    dense = DenseRetriever(units, MODELS.sentence_bert, dev)
    retrievers = [bm25, dense, HybridRetriever(bm25, dense)]
    s2 = os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")
    if os.path.exists(s2):
        retrievers.append(FusionRetriever(units, s2, dev))
    lat = {}
    for r in retrievers:
        r.search(queries[0]["query"], 10)  # warm-up
        times = []
        for q in queries:
            t = time.perf_counter()
            r.search(q["query"], 10)
            times.append((time.perf_counter() - t) * 1000)
        lat[r.name] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    out["query_latency"] = lat
    out["corpus_units"] = len(units)
    path = os.path.join(PATHS.results_dir, "runtime.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("Saved", path)


if __name__ == "__main__":
    main()
