# ConceptBound RAG evaluation

Run: 2026-11-11T23:23:30 | index: 157 segments from 5 lectures | questions: {'in_scope': 66, 'out_of_scope': 10, 'human_written': 17}

All numbers below come from running `python -m eval.run` on this index. Drafted questions were written from the transcripts and still need manual verification (see eval/README.md).

## 1. Retrieval ablation (in-scope questions)

### All in-scope questions

| System | Recall@5 | Recall@10 | MRR [95% CI] | nDCG@10 | latency p50 / p95 (ms) | n |
|---|---|---|---|---|---|---|
| BM25 only | 0.637 | 0.743 | 0.613 [0.52, 0.70] | 0.588 | 7 / 10 | 66 |
| Dense (Sentence-BERT) only | 0.576 | 0.714 | 0.549 [0.45, 0.65] | 0.544 | 12 / 14 | 66 |
| Dense (fusion encoder) only | 0.221 | 0.379 | 0.202 [0.13, 0.28] | 0.226 | 12 / 16 | 66 |
| Hybrid (BM25 + Sentence-BERT) | 0.673 | 0.764 | 0.633 [0.54, 0.72] | 0.611 | 12 / 14 | 66 |
| Current app mode ('auto') | 0.673 | 0.764 | 0.633 [0.54, 0.72] | 0.611 | 19 / 23 | 66 |
| Hybrid + reranker, OCR-first input (original) | 0.524 | 0.686 | 0.556 [0.46, 0.66] | 0.535 | 997 / 1039 | 66 |
| Hybrid + reranker, transcript-first input | 0.642 | 0.777 | 0.666 [0.58, 0.76] | 0.623 | 878 / 1039 | 66 |

### Drafted questions

| System | Recall@5 | Recall@10 | MRR [95% CI] | nDCG@10 | latency p50 / p95 (ms) | n |
|---|---|---|---|---|---|---|
| BM25 only | 0.656 | 0.753 | 0.618 [0.51, 0.72] | 0.590 | 7 / 10 | 49 |
| Dense (Sentence-BERT) only | 0.622 | 0.765 | 0.583 [0.47, 0.70] | 0.580 | 12 / 14 | 49 |
| Dense (fusion encoder) only | 0.266 | 0.431 | 0.234 [0.15, 0.33] | 0.263 | 12 / 16 | 49 |
| Hybrid (BM25 + Sentence-BERT) | 0.700 | 0.783 | 0.635 [0.52, 0.74] | 0.618 | 12 / 14 | 49 |
| Current app mode ('auto') | 0.700 | 0.783 | 0.635 [0.52, 0.74] | 0.618 | 19 / 23 | 49 |
| Hybrid + reranker, OCR-first input (original) | 0.551 | 0.671 | 0.573 [0.44, 0.70] | 0.536 | 997 / 1039 | 49 |
| Hybrid + reranker, transcript-first input | 0.705 | 0.783 | 0.716 [0.61, 0.81] | 0.657 | 878 / 1039 | 49 |

### Human-written pilot questions

| System | Recall@5 | Recall@10 | MRR [95% CI] | nDCG@10 | latency p50 / p95 (ms) | n |
|---|---|---|---|---|---|---|
| BM25 only | 0.582 | 0.717 | 0.596 [0.40, 0.79] | 0.583 | 7 / 10 | 17 |
| Dense (Sentence-BERT) only | 0.443 | 0.568 | 0.450 [0.26, 0.65] | 0.442 | 12 / 14 | 17 |
| Dense (fusion encoder) only | 0.093 | 0.227 | 0.108 [0.05, 0.18] | 0.118 | 12 / 16 | 17 |
| Hybrid (BM25 + Sentence-BERT) | 0.597 | 0.707 | 0.630 [0.44, 0.82] | 0.590 | 12 / 14 | 17 |
| Current app mode ('auto') | 0.597 | 0.707 | 0.630 [0.44, 0.82] | 0.590 | 19 / 23 | 17 |
| Hybrid + reranker, OCR-first input (original) | 0.444 | 0.730 | 0.508 [0.31, 0.71] | 0.529 | 997 / 1039 | 17 |
| Hybrid + reranker, transcript-first input | 0.460 | 0.760 | 0.521 [0.35, 0.69] | 0.527 | 878 / 1039 | 17 |

## 2. Refusal gate (retrieval confidence)

- **without reranker** (confidence source: dense): threshold 0.320 chosen on 30 dev questions (balanced accuracy 1.000). On 29 held-out questions: balanced accuracy **0.875**, out-of-scope refused **0.750**, in-scope wrongly refused 0.000.
- **with reranker** (confidence source: rerank): threshold 0.001 chosen on 30 dev questions (balanced accuracy 1.000). On 29 held-out questions: balanced accuracy **1.000**, out-of-scope refused **1.000**, in-scope wrongly refused 0.000.

## 3. Generation

| Metric | Value |
|---|---|
| Citation accuracy (answers citing a gold segment) | 1.000 (6/42 answered) |
| Refusal accuracy on out-of-scope questions | 1.000 |
| In-scope questions wrongly refused | 0.048 |
| Faithfulness (LLM-as-judge, 3 answers) | 0.733 |
| Hallucinated citations dropped by validation | 0 |
| LLM errors after retries / rate-limit retries / judge errors | 34 / 16 / 3 |

Model: groq:openai/gpt-oss-20b. `llm_request_ms` is the model's own response time; `llm_ms` also includes any waiting for the provider's rate limit.

| Stage | p50 (ms) | p95 (ms) |
|---|---|---|
| retrieval_ms | 23 | 62 |
| rerank_ms | 942 | 2147 |
| llm_request_ms | 1603 | 2153 |
| llm_ms | 183358 | 185659 |
| total_ms | 184110 | 186166 |

## Notes

- Gold segments are recomputed from answer spans for the current index (eval/gold.py).
- Drafted questions are paraphrased from the transcripts, which favours lexical retrieval; the human-written subset is the more realistic signal but is small (wide confidence intervals).
- The two reranker rows differ only in input order (OCR text first vs transcript first). The transcript-first fix was found by diagnosing the original result on this same question set, so its gain is optimistic until confirmed on new questions.
- Latencies are CPU-only on the development laptop.
