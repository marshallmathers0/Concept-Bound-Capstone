# LEARNING.md: how ConceptBound works, explained for interviews

Every section points at the file that implements it, so you can open the code while you revise. Numbers quoted at the end come from `reports/eval_results.md` (produced by `python -m eval.run`); never quote numbers that aren't in that file.

---

## 1. The big picture in one paragraph

ConceptBound turns hour-long lecture recordings into something you can *ask*. Offline, it:
1. extracts what was **said** (Whisper ASR), what was **on the slide** (EasyOCR on the auto-detected projector screen) and what the frame **looks like** (CLIP);
2. cuts each lecture into **concept-coherent segments** with a trained boundary detector;
3. embeds each segment and indexes it (FAISS + BM25).

Online, a question goes through **hybrid retrieval → cross-encoder reranking → a confidence gate → an LLM that may only use the retrieved segments → citation validation**, and comes back as an answer with clickable `[Lecture 3, 12:45]` links.

```
video ─► Whisper / OCR / CLIP ─► Stage 1 boundaries ─► segments ─► Stage 2 fusion + SBERT ─► FAISS + BM25   (offline)
question ─► rewrite follow-ups ─► hybrid retrieval ─► rerank ─► confidence gate ─► LLM (JSON) ─► validate citations ─► answer   (online)
```

---

## 2. Feature extraction (`src/feature_extraction/`)

- **Whisper ASR** (`asr_extractor.py`). It transcribes speech into timestamped segments. Two real problems were found and fixed in this project:
  - **Hallucination on silence.** Whisper invents text such as "I don't know. I don't know." during room noise. Fix: confidence filters (`no_speech_prob`, `avg_logprob`, compression ratio), `condition_on_previous_text=False`, and removal of short phrases repeated three or more times in 60 s.
  - **Wrong language.** Whisper auto-detects the language from the first 30 s, which in a classroom is often noise. Three lectures were decoded as **Welsh**. Fix: force `language="en"` and warn when a transcript doesn't look English.
- **OCR** (`ocr_extractor.py`). The projector screen is found automatically: the per-pixel *median* of many frames removes the moving instructor, Otsu thresholding finds the bright region, and the largest screen-shaped blob is kept. Cropping to it also removes the burnt-in CCTV clock, which would otherwise look like a "slide change" every 3 seconds.
- **CLIP** gives a visual embedding per sampled frame (every 3 s).

**Interview point.** All three failures were *silent*: the code ran and produced numbers, but the numbers were wrong. The fix was to make the pipeline fail loudly (no empty or random fallbacks) and to add sanity checks.

---

## 3. Stage 1: Pedagogical Boundary Detector (`src/stage1_boundary/`)

**Goal.** Find where one concept ends and the next begins.

**Input.** Per 3-second step, three scalars plus a 32-d CLIP projection:
- slide-text change, `1 − similarity(OCR_t, OCR_{t−1})`
- topic drift, `1 − cos(SBERT(window_t), SBERT(window_{t−1}))`
- optical-flow magnitude

**Model.** A temporal encoder (BiLSTM by default; CNN and Transformer variants exist) outputs an L2-normalized embedding `z_t` for every step. The **boundary score** is `b_t = 1 − z_t · z_{t+1}`, high when neighbouring steps look different.

**Contrastive loss: how positive and negative pairs are formed** (`losses.py`):
1. **Adjacent margin loss.** For each neighbouring pair (t, t+1):
   - same segment (a *positive* pair): pull together, loss `(1 − cos)²`;
   - across a boundary (a *negative* pair): push apart, loss `max(0, cos − (1 − m))²` with margin `m = 0.5`.
2. **Segment InfoNCE.** For each step (the *anchor*), sample **one positive** from the same segment and **K = 4 negatives** from other segments. A softmax over the 5 similarities (temperature 0.1) is trained to pick the positive.

Total: `L = L_adj + 0.5 · L_NCE`.

**Where the labels come from.** Human boundaries when annotated; otherwise *pseudo-labels* (peaks of the input signals). Be honest in interviews: a model trained only on pseudo-labels can at best re-learn the heuristic, so the heuristic is always reported as a baseline.

**Post-processing** (`segment.py`). Peaks of `b_t` above mean + std, then non-maximum suppression so segments last 60–600 s.

---

## 4. Stage 2: Cross-Modal Segment Fusion Encoder (`src/stage2_retrieval/fusion_model.py`)

Each segment has three embeddings: slide text (SBERT), transcript (SBERT) and visual (mean CLIP).

**Cross-attention fusion** (`fusion_mode="cross_attention"`):
1. Project each modality to 384-d, z-score normalize, add a learned *modality-type* embedding. The tokens are now `[OCR] [ASR] [VIS]`.
2. One transformer block of **multi-head self-attention across the 3 tokens**, so each modality can attend to the others (e.g. the transcript token can pull in slide text). Then a feed-forward layer and LayerNorms.
3. Mean-pool the 3 tokens → MLP → L2-normalize → `z_seg`.

A **query projector** (2-layer MLP) maps the SBERT query embedding into the same space.

**Training.** Symmetric InfoNCE over (query, segment) pairs with **in-batch negatives**: in a batch of B pairs, each query's positive is its own segment and the other B−1 segments are negatives, and the same holds in the segment→query direction. Temperature τ = 0.07. Pairs that share the same target segment are *masked*, because they are not real negatives (false negatives would push identical segments apart).

**Variants** (the fusion ablation):
- `concat`: no attention.
- `pairwise`: adapted from Arora et al. (LREC-COLING 2024). Each modality first goes through its own *unimodal expert*; one *shared* transformer block then fuses each modality **pair**, and extra contrastive losses are added per pair.

**Honest caveat.** The checkpoint in this repo was trained on automatically generated pseudo-queries and before several data fixes. The eval shows how it compares; don't overclaim it.

---

## 5. Retrieval (`src/stage2_retrieval/retrieval.py`, `src/retrieval/`)

- **FAISS `IndexFlatIP`.** An *exact* inner-product search. Because the vectors are L2-normalized, inner product equals cosine similarity. *Why not an approximate index (IVF, HNSW)?* With a few hundred to a few thousand segments, exact search takes under a millisecond. Approximate indexes only pay off at roughly 10⁵ or more vectors, and they add tuning (nlist, efSearch) and recall loss.
- **BM25** (`rank_bm25.BM25Okapi`, tokenizer in `src/utils/text_search.py`). A lexical score: term frequency (saturating, parameter k1) × inverse document frequency, normalized by document length (b). Great for exact terms ("LSTM", "perplexity").
- **Dense retrieval** (Sentence-BERT `all-MiniLM-L6-v2`). Query and segment are embedded separately, scored by cosine. It handles paraphrases ("remember far-back words" ≈ "long-term dependencies"). Segment texts longer than 256 tokens are split into chunks and the chunk embeddings averaged, because truncating would encode only the first ~20 seconds of a segment.
- **BM25 vs dense.** BM25 misses synonyms but is precise on rare terms, needs no training and is explainable. Dense handles meaning but can drift to vaguely related text, and struggles with rare technical tokens and ASR misspellings. They fail differently, which is why hybrid works.
- **How hybrid scores are combined** (`_combined_fallback_search`). Take the top 2k from each retriever; normalize BM25 scores by dividing by the max (so they lie in 0–1; cosine already does); final score = 0.5 · BM25_norm + 0.5 · cosine.
- **The existing "auto" mode** (dynamic confidence thresholding). It uses the fusion model's FAISS scores, but if they look degenerate (mean < 0.15, mean > 0.85 or variance < 0.01, typical of an untrained or collapsed model), it falls back to the hybrid. Otherwise it blends 0.7 · FAISS + 0.3 · text similarity.

### Why a reranker helps (`src/retrieval/reranker.py`)
- First-stage retrievers are **bi-encoders**: the query and segment never "see" each other.
- A **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) reads `[query] [SEP] [segment]` together, so attention can match a specific query word to a specific passage word. It is much more accurate, but it must run once per pair, which is too slow for the whole corpus.
- So: retrieve the top 20 cheaply, rerank those 20 accurately. The costs are latency (it is the slowest retrieval stage on CPU, as the eval shows) and that it was trained on web search (MS MARCO), not lectures.

---

## 6. The RAG layer (`src/rag/`)

| Step | File | What it does / why |
|---|---|---|
| Follow-up rewrite | `rewrite.py` | "explain it more simply" is short or has a pronoun, so the last 2 user questions are prepended before retrieval. Deterministic; optional LLM rewrite (`RAG_LLM_REWRITE=1`). |
| Retrieval + rerank | `src/retrieval/search.py` | Reuses the existing retriever; adds reranking and a confidence score. |
| **Confidence gate** | `pipeline.py` | If confidence is below the threshold, answer "This isn't covered in the indexed lectures" **without calling the LLM**. Confidence is the reranker probability for the top hit, or, without the reranker, the best cosine similarity in the whole collection. The threshold is *calibrated* on dev questions (in-scope vs out-of-scope) by `eval/run.py` and evaluated on held-out ones. |
| Context builder | `context.py` | Removes overlapping segments, labels them `[S1]…[S5]`, keeps within a token budget (1,800 tokens, about 450 per segment; estimate = words × 1.3). |
| Prompt | `prompts.py` | "Use ONLY these segments, cite every factual sentence with [S#], set covered=false if they don't contain the answer, return JSON." |
| Structured output | `schema.py` | Pydantic `LLMAnswer{answer, citations, covered}`. The parser strips ```fences```, finds the first JSON object, coerces types, and returns an error instead of crashing. |
| **Citation validation** | `citations.py` | Drops cited labels that were never in the context; removes sentences supported *only* by fake citations; turns valid labels into `[Lecture 3, 12:45]` links (`#t=765` for files, `&t=765s` for YouTube). If no valid citation remains, the answer is refused. |
| LLM providers | `src/llm/` | One `LLMClient.complete(system, user)` interface; Groq, Gemini and Ollama over plain HTTP; timeout, retries with exponential backoff and jitter, `Retry-After`; clear errors for a missing key or rate limit. |

### How RAG reduces hallucination
- The model answers from retrieved evidence instead of its memory (*grounding*).
- It is told to say "not covered" instead of guessing.
- The **gate** stops obviously off-topic questions before the LLM sees them.
- **Citation validation** means every surviving claim points at a real moment a student can check.

It does **not** make hallucination impossible. The LLM can still misread a segment, which is what the faithfulness metric measures.

### Why short labels `[S1]` instead of raw segment IDs?
LLMs copy short tokens far more reliably than IDs like `college_lec03#012`. The pipeline maps labels back to IDs, so the cited ID is still exact and checkable.

---

## 7. Evaluation (`eval/`)

- **Gold.** Every in-scope question stores its answer as a time span; `eval/gold.py` turns it into segment IDs of the current index. A segment is gold if it overlaps the span by at least half its own length, or covers at least half the span.
- **Recall@k.** The fraction of a question's gold segments found in the top k.
- **MRR.** 1 / rank of the first gold hit (0 if none in the top 10).
- **nDCG@10.** Rewards gold segments ranked higher (gain / log2(rank+1)), normalized by the ideal ranking.
- **Refusal.** The threshold is chosen on *dev* questions to maximize balanced accuracy (answering in-scope and refusing out-of-scope count equally), then reported on *test* questions. This keeps the threshold from being tuned on the numbers you report.
- **Generation** (needs an LLM key):
  - **Citation accuracy:** the fraction of answered questions that cite at least one gold segment.
  - **Faithfulness:** an LLM-as-judge splits the answer into claims and checks each against only the *cited* segments.
  - **Refusal accuracy:** measured end to end.
  - **Latency:** p50 and p95 per stage.

**Limitations of LLM-as-judge.** It is lenient towards fluent text; it may share blind spots with the generator if they're the same model family; claim splitting is inconsistent; it needs an API call per answer; and it only checks support, not whether the answer is *complete*. Spot-check 10–20 answers by hand and report the agreement.

---

## 8. 25 likely interview questions (with answers grounded in this code)

1. **What does ConceptBound do, in one line?** It answers questions about recorded lectures with answers grounded in the lectures and cited to the exact moment (`[Lecture 3, 12:45]`), and refuses when the lectures don't cover the question.
2. **Why segment by concept instead of fixed 60-second windows?** A fixed window can cut an explanation in half or mix two topics, which hurts both retrieval and the context the LLM reads. The project measures this directly (concept segments vs fixed windows, including a control with the same average length) instead of assuming it.
3. **How are positive and negative pairs formed in Stage 1?** Neighbouring steps in the same segment are positives and neighbours across a boundary are negatives (margin loss). For InfoNCE, each step is an anchor with one positive from its segment and four negatives from other segments.
4. **What does the margin do?** Boundary-crossing neighbours are only penalized while their cosine similarity is above 1 − m = 0.5. Once they're "different enough", the loss stops pushing, which avoids wasting capacity.
5. **What is InfoNCE in Stage 2, and what are in-batch negatives?** A softmax cross-entropy where the correct segment must score higher than every other segment in the batch. The other segments in the batch are the negatives, so no explicit negative mining is needed.
6. **Why mask same-segment pairs in the InfoNCE?** Two queries can target the same segment. Treating that segment as a negative for the other query would push identical items apart (false negatives).
7. **What is cross-attention doing over only 3 tokens?** Each modality token attends to the others, so the model learns per segment how much to trust slide text vs speech vs visuals. Concatenation would apply one fixed mixing.
8. **Why FAISS IndexFlatIP?** Exact search; inner product equals cosine on normalized vectors; at this scale it is sub-millisecond, so approximate indexes would add error and tuning for no gain.
9. **BM25 vs dense retrieval?** BM25 matches exact terms (precise on jargon, no training, explainable) but misses paraphrases. Dense matches meaning but can drift and is weaker on rare tokens. Hybrid covers both failure modes.
10. **How do you combine BM25 and dense scores?** BM25 is max-normalized to 0–1 and averaged 50/50 with cosine over the union of both top-k lists. A known alternative is reciprocal rank fusion, which ignores scale entirely; the research harness uses RRF.
11. **Why add a reranker if hybrid already works?** Bi-encoders never compare query and passage jointly. A cross-encoder does, so it fixes the ordering of near-misses. It costs CPU latency, measured in the eval.
12. **How do you stop the LLM from making things up?**
    - Context-only prompt with a "covered" flag.
    - A refusal gate before the LLM.
    - JSON output validated by Pydantic.
    - Citation validation that drops invalid sources and the sentences that relied on them.
13. **How is the refusal threshold chosen?** On dev questions (in-scope plus out-of-scope), maximizing balanced accuracy, then reported on held-out questions. It isn't hand-picked and isn't tuned on the reported numbers.
14. **What if the LLM returns broken JSON?** `parse_llm_output` strips fences, finds the first JSON object and validates it. On failure the API returns the top lecture moments with `mode="llm_error"` instead of crashing. This is tested.
15. **What if the LLM cites a segment that doesn't exist?** It's dropped, sentences that depended only on it are removed, and if nothing valid remains the answer becomes a refusal. This is tested with fake labels.
16. **How do follow-up questions work?** If the new question is short or uses a pronoun or "more/again", the last two user questions are prepended before retrieval. It is deterministic and testable; an LLM rewrite is optional.
17. **Why build the index offline?** Whisper and OCR take minutes per lecture on CPU. The app loads precomputed FAISS, metadata and embeddings, so it starts in seconds and fits the free CPU tier.
18. **How do you switch LLM providers?** A small `LLMClient` interface; `LLM_PROVIDER=groq|gemini|ollama` chooses the implementation; keys only come from environment variables or Space secrets.
19. **How do you handle rate limits?** Retries with exponential backoff and jitter, honouring `Retry-After`, then a clear `RateLimitError` the UI shows. The demo keeps working in retrieval-only mode.
20. **How do you evaluate retrieval?** Recall@5/10, MRR and nDCG@10 against gold segments, as an ablation: BM25, dense, fusion, hybrid, auto, and hybrid + reranker. Human-written questions are reported separately because drafted questions favour keyword search.
21. **How do you evaluate generation?** Citation accuracy, faithfulness via LLM-as-judge (with its limitations), refusal accuracy on out-of-scope questions, and p50/p95 latency per stage.
22. **What was the hardest bug?** Silent data failures:
    - OCR silently returned nothing for every lecture;
    - transcripts were repeated about 10×, so embeddings only saw a segment's first seconds;
    - Whisper decoded three lectures as Welsh.

    The lesson: validate data, not just code, and make pipelines fail loudly.
23. **Why short `[S#]` labels in the prompt?** LLMs copy short labels reliably; the code maps them back to exact segment IDs.
24. **What would you change at 10,000 lectures?** An approximate FAISS index (HNSW or IVF-PQ); a persisted BM25 or a search engine (OpenSearch); embedding caching; a GPU reranker or a smaller one; async LLM calls; and a proper human-labelled evaluation set.
25. **What are the limitations?** A small evaluation set from one course; drafted questions that still need human verification; a fusion encoder trained on weak pseudo-queries; an LLM judge with known biases; and recordings that can't be published without consent, so the public demo needs openly licensed lectures.

---

## 9. Honest limitations and next steps

- **Small evaluation.** A few lectures from one course; the confidence intervals in the report are wide.
- **Drafted questions are unverified** and favour BM25. The human-written subset is the more honest signal.
- **The fusion encoder** was trained on pseudo-queries before the data fixes; retraining on human queries is the obvious next step.
- **LLM-as-judge bias** (see section 7). Validate it with a manual spot check.
- **Privacy.** The recordings include students (one lecture contains a roll call with names). Use `lecture_span` annotations to exclude such parts, and use openly licensed lectures for any public demo.
- **Next:**
  - a human-verified evaluation set of 100+ questions;
  - retraining Stage 2 on real questions;
  - streaming LLM answers;
  - a small user study measuring how long students take to find an answer.

---

## 10. Measured results (from `reports/eval_results.md`; quote only these)

Index: 157 segments from 5 one-hour lectures. Questions: 66 in-scope (49 drafted from transcripts, not yet human-verified; 17 human-written) and 10 out-of-scope.

| System (all 66 questions) | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|
| BM25 only | 0.637 | 0.743 | 0.613 | 0.588 | 8 ms |
| Dense (Sentence-BERT) only | 0.576 | 0.714 | 0.549 | 0.544 | 12 ms |
| Dense (fusion encoder) only | 0.221 | 0.379 | 0.202 | 0.226 | 12 ms |
| Hybrid (BM25 + SBERT) | 0.673 | 0.764 | 0.633 | 0.611 | 12 ms |
| Hybrid + reranker (OCR-first input, original) | 0.524 | 0.686 | 0.556 | 0.535 | 2.5 s |
| Hybrid + reranker (transcript-first input) | 0.642 | 0.777 | 0.666 | 0.623 | 1.8 s |

- **Human-written subset (17 questions).** Hybrid MRR 0.630; reranker 0.521; BM25 0.596. The reranker does **not** help on real student wording.
- **Refusal gate on 29 held-out questions:** balanced accuracy 1.000 with reranker confidence, 0.875 with dense confidence. Only a handful of the held-out questions are out of scope, so this is a weak estimate.
- **Generation metrics:** not run (no API key available). Don't claim any faithfulness or citation-accuracy number until `python -m eval.run` has produced it.

**How to talk about the reranker result (good interview story).** The reranker first made results *worse*. Diagnosis: the cross-encoder reads at most 512 tokens, and the code put noisy OCR text first, so it mostly saw garbage. Putting the transcript first raised MRR from 0.556 to 0.666. The fix was found on the same question set, though, and on the human-written questions the reranker still underperforms hybrid. So the honest conclusion is "keep it switchable and validate on new human questions", not "the reranker is better".

## 11. Resume bullets (real numbers only)

- Built **ConceptBound**, a RAG system that answers questions about hour-long lecture videos with timestamp-cited answers (Whisper, EasyOCR, CLIP, FAISS, BM25, Sentence-BERT, cross-encoder, FastAPI, Gradio). It indexes 157 concept segments from 5 lectures and refuses out-of-scope questions with a calibrated confidence gate (balanced accuracy 1.00 on 29 held-out questions).
- Designed a retrieval ablation over 66 annotated questions (Recall@k, MRR, nDCG@10). Hybrid BM25 + dense retrieval reached MRR 0.633 vs 0.613 (BM25) and 0.549 (dense). Diagnosed a cross-encoder input-ordering bug that raised reranker MRR from 0.556 to 0.666.
- Found and fixed silent data bugs that corrupted 3 of 7 processed lectures (Whisper decoding English lectures as Welsh) and every OCR signal (silent fallback to an empty backend). Added fail-loud checks and a 25-test pytest suite with a mocked LLM for citation validation, refusal and malformed-output handling.
