"""
Everything after feature extraction, in order, ending with a rebuilt paper.

    python scripts/run_paper_pipeline.py                # full run
    python scripts/run_paper_pipeline.py --eval-only    # after new annotations: skip training

Steps:
  1. train Stage 1               6. build FAISS index (demo app)
  2. segment all lectures        7. evaluation on the test split (+ re-ranker)
  3. per-segment features        8. all ablations
  4. pseudo-queries (training)   9. runtime / latency measurement
  5. train Stage 2              10. collect results -> paper (guide + blind)

Every step's output is logged to logs/pipeline_<step>.log; the run stops at the
first failure.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def step(name, cmd, log_dir):
    print(f"\n=== {name} ===\n  $ {' '.join(cmd)}", flush=True)
    t = time.time()
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONIOENCODING": "utf-8"}
    with open(os.path.join(log_dir, f"pipeline_{name}.log"), "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"  -> {'ok' if r.returncode == 0 else 'FAILED'} in {(time.time() - t) / 60:.1f} min "
          f"(log: logs/pipeline_{name}.log)", flush=True)
    if r.returncode != 0:
        sys.exit(f"Step '{name}' failed; see logs/pipeline_{name}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true", help="Reuse checkpoints; only evaluate, ablate (no retraining) and rebuild the paper.")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    t0 = time.time()

    if not args.eval_only:
        step("stage1_train", [PY, "-m", "src.stage1_boundary.train", "--epochs", "30"], logs)
        step("segment", [PY, "scripts/segment_videos.py"], logs)
        step("segment_features", [PY, "scripts/prepare_segment_features.py"], logs)
        step("pseudo_queries", [PY, "scripts/generate_pseudo_queries.py"], logs)
        step("stage2_train", [PY, "-m", "src.stage2_retrieval.train", "--epochs", "20"], logs)
        step("index", [PY, "-m", "src.stage2_retrieval.index_builder"], logs)
    step("validate_annotations", [PY, "scripts/validate_annotations.py"], logs)
    step("evaluate", [PY, "scripts/run_evaluation.py", "--split", args.split, "--rerank",
                      "--run-name", f"paper_{args.split}"], logs)
    step("ablations", [PY, "scripts/run_ablations.py", "--split", args.split]
         + (["--skip-training"] if args.eval_only else []), logs)
    step("runtime", [PY, "scripts/measure_runtime.py"], logs)
    step("collect", [PY, "scripts/collect_paper_results.py"], logs)
    step("figure2", [PY, "paper/make_figure2.py", "college_lec03"], logs)
    step("paper_guide", ["node", "paper/build_paper.js"], logs)
    step("paper_blind", ["node", "paper/build_paper.js", "--blind"], logs)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Paper: paper/ICALT2027_draft_guide.docx / _blind.docx")


if __name__ == "__main__":
    main()
