"""
Run the whole RAG evaluation and write reports/eval_results.md (+ .json).

    python -m eval.run
    python -m eval.run --no-generation      # retrieval + refusal gate only

1. Retrieval ablation on every in-scope question (gold = segments covering the
   annotated answer span): BM25, dense Sentence-BERT, dense fusion encoder,
   hybrid (BM25 + Sentence-BERT), the app's current 'auto' mode, and
   hybrid + cross-encoder reranker. Recall@5/10, MRR, nDCG@10, latency.
   Reported for all questions and separately for human-written questions.
2. Refusal gate: the confidence threshold is calibrated on the 'dev' half of
   the drafted questions (in-scope + out-of-scope) and evaluated on 'test'.
   Saved to <index>/thresholds.json for the app.
3. Generation (only if an LLM is configured): citation accuracy, end-to-end
   refusal accuracy, faithfulness (LLM-as-judge), per-stage latency.
   If no LLM is configured this section says so; nothing is invented.
"""

import argparse
import datetime
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from eval.gold import dev_test_split, gold_segments, load_questions
from eval.metrics import bootstrap_ci, mean_metrics, percentiles, ranking_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "eval", "questions.jsonl")
REPORTS = os.path.join(ROOT, "reports")

SYSTEMS = [  # (label, retriever mode, rerank, reranker input order)
    ("BM25 only", "bm25", False, None),
    ("Dense (Sentence-BERT) only", "dense", False, None),
    ("Dense (fusion encoder) only", "model", False, None),
    ("Hybrid (BM25 + Sentence-BERT)", "hybrid", False, None),
    ("Current app mode ('auto')", "auto", False, None),
    ("Hybrid + reranker, OCR-first input (original)", "hybrid", True, "ocr_first"),
    ("Hybrid + reranker, transcript-first input", "hybrid", True, "transcript_first"),
]


def retrieval_ablation(search, qs):
    rows, lat = {}, {}
    for label, mode, rerank, order in SYSTEMS:
        per_q, times = [], []
        for q in qs:
            t = time.perf_counter()
            res = search.search(q["question"], top_k=10, mode=mode, rerank=rerank,
                                rerank_order=order or "transcript_first")
            times.append((time.perf_counter() - t) * 1000)
            per_q.append({**ranking_metrics([h["segment_id"] for h in res.hits], q["gold"]),
                          "id": q["id"], "source": q["source"]})
        rows[label] = per_q
        lat[label] = percentiles(times)
    return rows, lat


def summarize(per_q, source=None):
    sel = [r for r in per_q if source is None or r["source"] == source]
    if not sel:
        return None
    m = mean_metrics([{k: r[k] for k in ("R@5", "R@10", "MRR", "nDCG@10")} for r in sel])
    m["MRR_ci"] = bootstrap_ci([r["MRR"] for r in sel])
    m["n"] = len(sel)
    return m


def calibrate(search, qs, rerank):
    """Pick the confidence threshold that maximizes balanced accuracy
    (answer in-scope, refuse out-of-scope) on dev; report it on test."""
    scored = []
    for q in qs:
        res = search.search(q["question"], top_k=5, mode="hybrid", rerank=rerank)
        scored.append((q, res.confidence, res.confidence_source))
    source = scored[0][2]

    def bal_acc(items, th):
        ins = [c >= th for q, c, _ in items if q["in_scope"]]
        oos = [c < th for q, c, _ in items if not q["in_scope"]]
        if not ins or not oos:
            return None, None, None
        return (np.mean(ins) + np.mean(oos)) / 2, float(np.mean(oos)), float(1 - np.mean(ins))

    dev = [s for s in scored if dev_test_split(s[0]) == "dev"]
    test = [s for s in scored if dev_test_split(s[0]) == "test" and s[0]["source"] != "pilot_human"]
    cands = sorted(set(round(c, 3) for _, c, _ in dev))
    best = max(cands, key=lambda th: (bal_acc(dev, th)[0] or 0, -th))
    ba, oos_refused, ins_refused = bal_acc(test, best)
    return {"source": source, "threshold": best, "dev_balanced_acc": bal_acc(dev, best)[0],
            "test_balanced_acc": ba, "test_oos_refusal_rate": oos_refused, "test_inscope_false_refusal": ins_refused,
            "n_dev": len(dev), "n_test": len(test),
            "conf_in_scope": [round(c, 3) for q, c, _ in scored if q["in_scope"]],
            "conf_out_of_scope": [round(c, 3) for q, c, _ in scored if not q["in_scope"]]}


def generation_eval(pipeline, judge_llm, qs, pause=0.0):
    from eval.judge import judge_faithfulness
    out, lat = [], {"retrieval_ms": [], "rerank_ms": [], "llm_request_ms": [], "llm_ms": [], "total_ms": []}
    rate_limited = 0
    for i, q in enumerate(qs):
        if i and pause:
            time.sleep(pause)  # stay under the provider's tokens-per-minute limit
        r = pipeline.ask(q["question"])
        rate_limited += int(r.latency_ms.get("llm_rate_limited", 0))
        for k in lat:
            if k in r.latency_ms:
                lat[k].append(r.latency_ms[k])
        cited = {c.segment_id for c in r.citations}
        row = {"id": q["id"], "in_scope": q["in_scope"], "mode": r.mode, "refused": r.refused,
               # only genuine grounded answers count (not error fallbacks that list retrieved moments)
               "cites_gold": bool(cited & q["gold"]) if q["in_scope"] and r.mode == "rag" else None,
               "n_dropped_citations": len(r.dropped_citations), "error": r.error}
        if r.mode == "rag":
            hits = [pipeline.search.by_id[c.segment_id] for c in r.citations]
            try:
                row["faithfulness"] = judge_faithfulness(judge_llm, hits, r.answer)
            except Exception as e:
                row["faithfulness"] = None
                row["judge_error"] = str(e)
        out.append(row)
    ins = [r for r in out if r["in_scope"]]
    answered = [r for r in ins if r["mode"] == "rag"]
    oos = [r for r in out if not r["in_scope"]]
    faith = [r["faithfulness"] for r in out if r.get("faithfulness") is not None]
    return {
        "citation_accuracy": float(np.mean([r["cites_gold"] for r in answered])) if answered else None,
        "n_answered_in_scope": len(answered), "n_in_scope": len(ins),
        "refusal_accuracy_oos": float(np.mean([r["refused"] for r in oos])) if oos else None,
        "false_refusal_in_scope": float(np.mean([r["refused"] for r in ins])) if ins else None,
        "faithfulness": float(np.mean(faith)) if faith else None, "n_judged": len(faith),
        "dropped_citations_total": int(sum(r["n_dropped_citations"] for r in out)),
        "llm_errors": sum(r["mode"] == "llm_error" for r in out), "rate_limited_calls": rate_limited,
        "judge_errors": sum(1 for r in out if r.get("judge_error")),
        "latency": {k: percentiles(v) for k, v in lat.items() if v}, "rows": out,
    }


def fmt(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_report(res, path_md):
    L = [f"# ConceptBound RAG evaluation", "",
         f"Run: {res['run_at']} | index: {res['index']['num_segments']} segments from "
         f"{len(res['index']['videos'])} lectures | questions: {res['counts']}", "",
         "All numbers below come from running `python -m eval.run` on this index. Drafted questions "
         "were written from the transcripts and still need manual verification (see eval/README.md).", "",
         "## 1. Retrieval ablation (in-scope questions)", ""]
    for subset, title in ((None, "All in-scope questions"), ("drafted", "Drafted questions"),
                          ("pilot_human", "Human-written pilot questions")):
        L += [f"### {title}", "", "| System | Recall@5 | Recall@10 | MRR [95% CI] | nDCG@10 | latency p50 / p95 (ms) | n |",
              "|---|---|---|---|---|---|---|"]
        for label, *_ in SYSTEMS:
            m = summarize(res["retrieval"][label], subset)
            if not m:
                continue
            lat = res["retrieval_latency"][label]
            L.append(f"| {label} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.3f} "
                     f"[{m['MRR_ci'][0]:.2f}, {m['MRR_ci'][1]:.2f}] | {m['nDCG@10']:.3f} | "
                     f"{lat['p50']:.0f} / {lat['p95']:.0f} | {m['n']} |")
        L.append("")
    L += ["## 2. Refusal gate (retrieval confidence)", ""]
    for name, c in res["refusal"].items():
        L += [f"- **{name}** (confidence source: {c['source']}): threshold {c['threshold']:.3f} chosen on "
              f"{c['n_dev']} dev questions (balanced accuracy {fmt(c['dev_balanced_acc'])}). On {c['n_test']} held-out "
              f"questions: balanced accuracy **{fmt(c['test_balanced_acc'])}**, out-of-scope refused "
              f"**{fmt(c['test_oos_refusal_rate'])}**, in-scope wrongly refused {fmt(c['test_inscope_false_refusal'])}."]
    L += ["", "## 3. Generation", ""]
    g = res.get("generation")
    if not g:
        L += [f"**Not run:** {res.get('generation_skipped', 'no LLM configured')}. Faithfulness, citation "
              "accuracy, end-to-end refusal and LLM latency need a configured provider "
              "(set LLM_PROVIDER and an API key, then re-run `python -m eval.run`). No numbers are reported "
              "rather than invented ones."]
    else:
        L += ["| Metric | Value |", "|---|---|",
              f"| Citation accuracy (answers citing a gold segment) | {fmt(g['citation_accuracy'])} ({g['n_answered_in_scope']}/{g['n_in_scope']} answered) |",
              f"| Refusal accuracy on out-of-scope questions | {fmt(g['refusal_accuracy_oos'])} |",
              f"| In-scope questions wrongly refused | {fmt(g['false_refusal_in_scope'])} |",
              f"| Faithfulness (LLM-as-judge, {g['n_judged']} answers) | {fmt(g['faithfulness'])} |",
              f"| Hallucinated citations dropped by validation | {g['dropped_citations_total']} |",
              f"| LLM errors after retries / rate-limit retries / judge errors | {g['llm_errors']} / {g['rate_limited_calls']} / {g['judge_errors']} |", "",
              f"Model: {res.get('llm_model', '?')}. `llm_request_ms` is the model's own response time; `llm_ms` "
              "also includes any waiting for the provider's rate limit.", ""]
        L += ["| Stage | p50 (ms) | p95 (ms) |", "|---|---|---|"] + \
             [f"| {k} | {v['p50']:.0f} | {v['p95']:.0f} |" for k, v in g["latency"].items()]
    L += ["", "## Notes", "",
          "- Gold segments are recomputed from answer spans for the current index (eval/gold.py).",
          "- Drafted questions are paraphrased from the transcripts, which favours lexical retrieval; the human-written "
          "subset is the more realistic signal but is small (wide confidence intervals).",
          "- The two reranker rows differ only in input order (OCR text first vs transcript first). The "
          "transcript-first fix was found by diagnosing the original result on this same question set, so its gain "
          "is optimistic until confirmed on new questions.",
          "- Latencies are CPU-only on the development laptop."]
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--no-generation", action="store_true")
    ap.add_argument("--llm-pause", type=float, default=45.0,
                    help="Seconds between questions in the generation eval (Groq free tier: 8k tokens/min).")
    args = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)

    from src.llm.factory import get_llm_client
    from src.rag.pipeline import RAGPipeline
    from src.retrieval.search import SearchService

    search = SearchService(use_reranker=True)
    search.warm_up()
    qs = load_questions(args.questions)
    for q in qs:
        q["gold"] = gold_segments(q, search.retriever.metadata) if q["in_scope"] else set()
    missing = [q["id"] for q in qs if q["in_scope"] and not q["gold"]]
    if missing:
        print(f"WARNING: no gold segment in the index for {missing} (span outside indexed lectures?)")
    ins = [q for q in qs if q["in_scope"] and q["gold"]]

    res = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "index": json.load(open(os.path.join(search.index_dir, "index_info.json"))),
           "counts": {"in_scope": len(ins), "out_of_scope": sum(not q["in_scope"] for q in qs),
                      "human_written": sum(q["source"] == "pilot_human" for q in ins)}}
    print("Retrieval ablation ...", flush=True)
    res["retrieval"], res["retrieval_latency"] = retrieval_ablation(search, ins)

    print("Calibrating refusal thresholds ...", flush=True)
    gate_qs = [q for q in qs if q["source"] != "pilot_human" and (q["gold"] or not q["in_scope"])]
    res["refusal"] = {"without reranker": calibrate(search, gate_qs, rerank=False),
                      "with reranker": calibrate(search, gate_qs, rerank=True)}
    thresholds = {"dense": res["refusal"]["without reranker"]["threshold"],
                  "rerank": res["refusal"]["with reranker"]["threshold"]}
    json.dump(thresholds, open(os.path.join(search.index_dir, "thresholds.json"), "w"), indent=2)
    search.thresholds.update(thresholds)

    llm = None if args.no_generation else get_llm_client()
    if llm is None:
        res["generation_skipped"] = "--no-generation given" if args.no_generation else \
            "no LLM provider configured on this machine (no GROQ_API_KEY / GEMINI_API_KEY / Ollama)"
    else:
        print(f"Generation eval with {llm.name}:{llm.model} ...", flush=True)
        test_qs = [q for q in qs if dev_test_split(q) == "test" and (q["gold"] or not q["in_scope"])]
        res["llm_model"] = f"{llm.name}:{llm.model}"
        os.environ.setdefault("LLM_RETRIES", "6")
        res["generation"] = generation_eval(RAGPipeline(search, llm, cache_size=0), llm, test_qs, pause=args.llm_pause)

    def default(o):
        return sorted(o) if isinstance(o, set) else str(o)
    json.dump(res, open(os.path.join(REPORTS, "eval_results.json"), "w"), indent=2, default=default)
    write_report(res, os.path.join(REPORTS, "eval_results.md"))
    print(open(os.path.join(REPORTS, "eval_results.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
