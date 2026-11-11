# DEVLOG: ConceptBound

> The complete story of this project: what it does, how every part works and *why* it was built that way, how it was built day by day, what the measurements show, what went wrong and how it was fixed, and how to talk about it in an interview.
>
> Every number in this file comes from a file in the repo (mainly `reports/eval_results.md`). If a number isn't in a report, it isn't quoted here.

---

## Contents

- [Part I: The problem and the idea](#part-i-the-problem-and-the-idea)
- [Part II: How it works, end to end](#part-ii-how-it-works-end-to-end)
- [Part III: The A–Z of ConceptBound](#part-iii-the-az-of-conceptbound)
- [Part IV: How it was built, day by day](#part-iv-how-it-was-built-day-by-day)
- [Part V: Results](#part-v-results)
- [Part VI: Lessons, limitations and next steps](#part-vi-lessons-limitations-and-next-steps)
- [Part VII: Interview questions and answers](#part-vii-interview-questions-and-answers)
- [Part VIII: Two interview stories](#part-viii-two-interview-stories)
- [Appendix: command cheat sheet and file map](#appendix-command-cheat-sheet-and-file-map)

---

# Part I: The problem and the idea

## 1.1 The problem

Our department records every lecture with a fixed auditorium camera. Students use the recordings to revise, but a one-hour video is a terrible revision tool. A student who wants "the part where the professor explains why we use softmax" has to scrub through the timeline, guess, and scrub again.

Ordinary keyword search doesn't solve it either:

- **Student wording differs from lecture wording.** "Why can't an RNN remember far-back words?" is the same question as "long-term dependencies", but shares almost no words with it.
- **The evidence is spread across modalities.** Part of the answer is spoken, part is on the slide, and part is in a diagram.
- **Classroom recordings are messy.** The projector screen covers only part of a wide shot, a clock is burnt into the corner of every frame, there is dead air before class, and speech recognition makes mistakes on accented, spontaneous speech.

## 1.2 The idea

1. **Cut each lecture into concept-sized pieces** (not fixed 60-second windows), so each piece is one explanation.
2. **Describe each piece with everything we know about it**: what was said, what was on the slide, and what the frame looked like.
3. **Search those pieces with a strong hybrid retriever**, so a student's question finds the right moment in the right lecture.
4. **Answer in plain language with a large language model that may only use the retrieved pieces**, and cite them as clickable `[Lecture 3, 12:45]` links. If the lectures don't cover the question, say so instead of guessing.

## 1.3 Design principles

| Principle | What it means in practice |
|---|---|
| **Grounded, not generative** | The LLM only paraphrases retrieved lecture content; every claim needs a valid citation. |
| **Fail loudly** | No silent fallbacks to empty or random data; checks for wrong-language transcripts, empty OCR, and so on. |
| **Measure before claiming** | Every component is compared against simple baselines on human-annotated data; nothing is reported that wasn't measured. |
| **Offline heavy, online light** | Whisper, OCR and CLIP run once while indexing; the app only loads precomputed files and runs on a free CPU. |
| **Explainable choices** | Standard, well-understood techniques (BM25, Sentence-BERT, FAISS, cross-encoders, InfoNCE) over exotic ones. |
| **Privacy first** | Classroom video shows students; transcripts and indexes stay out of git; public demos use openly licensed lectures only. |

---

# Part II: How it works, end to end

```
OFFLINE (once per lecture)
 video ─┬─► Whisper ASR ──────────────► transcript segments (start, end, text)
        ├─► screen detection + EasyOCR ► slide text per 3 s step
        └─► CLIP + optical flow ───────► visual embedding + visual-change per step
                     │
                     ▼
        per-step signal stream (every 3 s)
                     │
                     ▼
        Stage 1: Pedagogical Boundary Detector ─► concept segments (60–600 s)
                     │
                     ▼
        per-segment text + 3 modality embeddings
          ├─► Stage 2 fusion encoder ─► FAISS index
          ├─► Sentence-BERT ──────────► cached dense embeddings
          └─► text ───────────────────► BM25 corpus + metadata JSON

ONLINE (every question)
 question (+ chat history)
   ─► follow-up rewrite ─► hybrid retrieval (BM25 + dense) ─► cross-encoder rerank
   ─► confidence gate ──(low)──► "This isn't covered in the indexed lectures."
                      └─(ok)──► context builder [S1..S5] ─► LLM (JSON) ─► Pydantic parse
   ─► citation validation ─► answer with [Lecture, mm:ss] links + latency
```

## 2.1 The data

- **11 recordings** of an undergraduate course on large language models, each exactly 64 minutes, 1080×608 at 25 fps, from one fixed auditorium camera.
- **What the camera sees:**
  - the projected slides in the left half of the frame
  - the stage and the instructor
  - part of the audience
  - burnt-in overlays: a clock in the top right and watermarks at the bottom
- **Quirks that matter:**
  - **Dead air:** pre-class chatter, breaks, and in one file about 37 minutes of empty stage after class.
  - **A roll call:** one lecture includes students' names being read out.
  - **Changing framing:** camera framing changes between sessions.
- **Sampling:** every recording is sampled every **3 seconds**, giving **1,281 steps per lecture**.

## 2.2 Feature extraction (`src/feature_extraction/`)

### Speech: Whisper (`asr_extractor.py`)
- **Model.** Whisper `base`, which runs on a laptop CPU. It outputs timestamped segments.
- **Forced English.** `language="en"`. *Why:* Whisper guesses the language from the first 30 s of audio, which in a classroom is noise. It decoded three English lectures as Welsh (see Story 1 in Part VIII).
- **Hallucination filters.** Whisper invents text over silence ("I don't know. I don't know."). We drop:
  - segments with high `no_speech_prob` and low `avg_logprob`;
  - segments with a high compression ratio (repetitive text);
  - short phrases repeated three or more times within 60 s;
  - and we set `condition_on_previous_text=False`, so one hallucination doesn't seed the next.
- **Sanity check.** `english_share()` warns if fewer than 50% of segments look like English.
- **Topic drift.** For every step, the transcript in a 30-second window is embedded with Sentence-BERT. *Topic drift* = 1 − cosine(window_t, window_{t−1}). A cheap TF-IDF pre-filter skips the encoder on stable stretches.

### Slides: EasyOCR on the auto-detected screen (`ocr_extractor.py`)
- **Screen detection.** Take the **per-pixel median** of many frames, which removes the moving instructor. Then Otsu-threshold it and keep the largest bright, screen-shaped blob. This runs per 5-minute chunk, because framing can change mid-recording, and `data/ocr_regions.json` can override it manually.
- *Why crop:* without it, OCR reads the burnt-in clock, which changes every second and would look like a "slide change" at every step.
- **Speed.** OCR only runs when the screen changes (SSIM below 0.95), which skips about 80–90% of frames.
- **Signal.** *Slide-text change* = 1 − string similarity between consecutive OCR texts.
- **Fail loudly.** If EasyOCR can't load, the extractor raises an error. It used to silently return empty text (see Story 1).

### Visuals: CLIP + optical flow
- A CLIP ViT-B/32 embedding per sampled frame, projected to 32-d.
- Optical-flow magnitude gives a *visual-change* signal.

**Per-step feature vector:** `[slide-text change, topic drift, visual change, CLIP(32)]` = 35 numbers every 3 seconds.

## 2.3 Stage 1: Pedagogical Boundary Detector (`src/stage1_boundary/`)

**Goal.** Find where one concept ends and the next begins.

**Why not just threshold one signal?** Slide changes are reliable but rare (several ideas can share one slide), and topic drift reacts to every digression. Paper Fig. 2 shows this on a real lecture. A model can learn to combine all the signals in context.

**Model.** A temporal encoder (BiLSTM by default; CNN and Transformer variants) maps the 35-d sequence to L2-normalized embeddings `z_t`. The boundary score is

```
b_t = 1 − z_t · z_{t+1}        (high when neighbouring steps look different)
```

**Loss: how positive and negative pairs are formed** (`losses.py`):
1. **Adjacent margin loss.**
   - Neighbours *in the same segment* (positive pairs) are pulled together: `(1 − cos)²`.
   - Neighbours *across a boundary* (negative pairs) are pushed apart until cosine falls below `1 − m`, with margin `m = 0.5`: `max(0, cos − (1 − m))²`.
2. **Segment InfoNCE.** For each step (the anchor), take **1 positive** from the same segment and **K = 4 negatives** from other segments. A softmax over the 5 similarities (temperature 0.1) must pick the positive. This gives a non-local signal beyond immediate neighbours.

Total: `L = L_adj + 0.5 · L_NCE`.

**Labels.** Human boundary annotations where available; otherwise *pseudo-labels* (peaks of the input signals). Honesty rule: a model trained only on pseudo-labels can at best re-learn the heuristic, so the heuristic is always reported as a baseline. A gold file with queries but no boundaries is *not* treated as "this lecture has no boundaries" (that was a real bug, fixed).

**Post-processing** (`segment.py`). Peaks above mean + std, then **non-maximum suppression** so segments last **60–600 s**. *Why:* the first version produced 12-second "concepts" next to 27-minute ones.

**Signal-removal ablation.** `--drop-signals ocr|drift|visual|clip` zeroes input channels. The mask is saved in the checkpoint, so inference applies the same mask.

## 2.4 Stage 2: Cross-Modal Segment Fusion Encoder (`src/stage2_retrieval/fusion_model.py`)

Each segment gets three embeddings:
- **slide text:** Sentence-BERT over the *distinct* slide texts shown in the segment;
- **transcript:** Sentence-BERT over the Whisper segments in the segment. Long texts are **chunked and averaged**, never truncated (truncation used to encode only the first ~20 s);
- **visual:** the mean CLIP embedding.

**Three fusion modes** (the fusion ablation):

| Mode | How | Why it exists |
|---|---|---|
| `cross_attention` (default) | The three tokens `[OCR][ASR][VIS]` get modality-type embeddings and pass through one transformer block (multi-head self-attention + FFN), then mean-pool → MLP → L2-normalize. | Lets the model decide *per segment* which modality to trust. |
| `concat` | Concatenate the projections, then MLP. | Baseline: no learned interaction. |
| `pairwise` | Adapted from Arora et al. (LREC-COLING 2024), the project's base paper. Each modality first goes through its own *unimodal expert*; one **shared** transformer block fuses each modality **pair**; pooled; plus a contrastive loss per pair. | Tests the base paper's design in the lecture domain. |

A **query projector** (2-layer MLP) maps the Sentence-BERT query embedding into the same space.

**Training: symmetric InfoNCE with in-batch negatives.**
- In a batch of B (query, segment) pairs, each query must score its own segment above the other B−1 segments, and vice versa. Temperature τ = 0.07.
- Pairs that share a target segment are **masked** (they aren't real negatives).
- Without human questions, training uses **pseudo-queries**: each segment's most informative sentence, its top keywords, and its slide title. These are **never** used for evaluation, because they're built from the segment's own text, so finding them is trivial.

**Honest status.** The checkpoint in the repo was trained on pseudo-queries before several data fixes. It's weak (MRR 0.20, see Part V), and retraining on human questions is the top item on the to-do list.

## 2.5 Indexing (`src/rag/build_index.py`)

```
python -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05
```

For each lecture: reuse the extracted features (or extract them with `--extract-missing`), run Stage 1, build the segment texts and embeddings, run Stage 2, then write:

| File | Content | Why |
|---|---|---|
| `segments.index` | FAISS `IndexFlatIP` over fusion embeddings | Exact cosine search (vectors are normalized) |
| `segments_metadata.json` | segment ID, video, title, URL, start/end, transcript, slide text | Everything needed for answers, citations and BM25 |
| `text_embeddings.npy` | Sentence-BERT embeddings of each segment's text | The app starts in seconds instead of ~30 s of encoding |
| `index_info.json` | build time, checkpoints, videos | Reproducibility |

Segments outside an annotated `lecture_span` are dropped, which removes pre-class chatter and the roll call. Current index: **157 segments from 5 lectures, built in about 63 s.**

## 2.6 Retrieval (`src/stage2_retrieval/retrieval.py`, `src/retrieval/`)

| Retriever | How it scores | Strong at | Weak at |
|---|---|---|---|
| **BM25** | term frequency (saturating, k1) × inverse document frequency, length-normalized (b) | exact technical terms, no training | synonyms and paraphrases |
| **Dense (Sentence-BERT `all-MiniLM-L6-v2`)** | cosine between query and segment embeddings | paraphrases, meaning | rare tokens, ASR misspellings |
| **Fusion (Stage 2)** | FAISS inner product with the learned encoder | in principle, multimodal evidence | currently weakly trained |
| **Hybrid** | top 2k from BM25 and dense; BM25 divided by its max (0–1); `0.5 · BM25 + 0.5 · cosine` | both failure modes covered | — |
| **`auto`** (the original app mode) | fusion scores, but falls back to hybrid if they look degenerate (mean < 0.15, > 0.85 or variance < 0.01); else 0.7 · fusion + 0.3 · text | robust to an untrained model | — |

**Cross-encoder reranker** (`src/retrieval/reranker.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`). Bi-encoders never look at the question and the segment *together*. A cross-encoder reads `[question][SEP][segment]` jointly and scores relevance much more precisely, but only for the top 20 candidates, because it's slow. **Input order matters:** the model reads at most 512 tokens, so we feed the transcript first and the noisy OCR text last (see Story 2).

**Confidence for the refusal gate** (`search.py`):
- *without reranker:* the best cosine similarity between the question and *any* segment in the collection;
- *with reranker:* the sigmoid of the top reranker logit.

## 2.7 The RAG pipeline (`src/rag/`)

1. **Follow-up rewrite** (`rewrite.py`). If a question is short or contains "it / this / more / again…", the last two user questions are prepended, so "explain it more simply" becomes searchable. It is deterministic; an LLM rewrite is optional.
2. **Retrieve and rerank** (above).
3. **Confidence gate.** If confidence is below the threshold, answer *"This isn't covered in the indexed lectures."* and **don't call the LLM**. The threshold is **calibrated** by `eval/run.py` on dev questions and checked on held-out ones, never hand-picked.
4. **Context builder** (`context.py`):
   - remove overlapping segments;
   - label the rest `[S1]…[S5]`;
   - stay inside a token budget: 1,800 tokens in total, about 450 per segment, estimated as words × 1.3.
5. **Prompt** (`prompts.py`). "Use ONLY these segments; cite every factual sentence as [S#]; if they don't contain the answer set `covered=false`; return JSON `{answer, citations, covered}`."
6. **Parse** (`schema.py`). Pydantic validation. A tolerant parser strips ```code fences```, finds the first JSON object and coerces types; on failure it returns an error instead of crashing.
7. **Validate citations** (`citations.py`):
   - drop labels the LLM cited that were never in its context;
   - remove sentences supported **only** by such fake labels;
   - turn valid labels into `[Lecture 3, 12:45]` links (`#t=765` for files, `&t=765s` for YouTube);
   - if no valid citation is left, refuse.
8. **Respond** with the answer, citations (video, timestamp, link, snippet), confidence, mode (`rag`, `retrieval_only`, `refused` or `llm_error`) and per-stage latency.

**No API key?** The pipeline runs in *retrieval-only* mode and shows the most relevant moments. **LLM down or rate-limited?** It returns those moments with a clear message.

### LLM providers (`src/llm/`)
- **Interface.** One method, `LLMClient.complete(system, user, json_mode)`.
- **Implementations:** **Groq** (OpenAI-compatible API; default `openai/gpt-oss-20b`, auto-switches if Groq retires a model), **Gemini** (REST), **Ollama** (local) and a **Mock** for tests.
- **Selection.** `LLM_PROVIDER` picks one; keys come only from environment variables (`.env`, Space secrets).
- **Robustness.** `post_json()` adds a timeout, retries 429/5xx/timeouts with **exponential backoff and jitter**, honours `Retry-After`, and turns failures into clear exceptions (`MissingAPIKeyError`, `RateLimitError`, `LLMTimeoutError`).

## 2.8 Serving

- **FastAPI** (`app/backend/main.py`):
  - `POST /ask`: answer, citations, confidence and latency.
  - `GET /health`: whether the index and LLM are ready.
  - The original `GET /api/search`, `/api/health` and `/api/stats` are unchanged, and the raw videos are served under `/videos`.
- **Gradio demo** (`space/app.py`):
  - question box with example questions;
  - answer and citation cards, each with an embedded player that starts at the cited second (YouTube iframe or local video);
  - a "Show retrieved segments" table with scores;
  - a reranker on/off switch;
  - a latency line;
  - follow-up questions.
- **Hugging Face Space** (`space/prepare_space.py`). Copies only the inference code, index, checkpoint and pinned requirements (about 9 MB), and **refuses to package lectures marked internal**.

## 2.9 Evaluation (`eval/`, `scripts/run_evaluation.py`)

- **Question set** (`eval/make_questions.py`):
  - 49 *drafted* questions (written from the transcripts; must be human-verified);
  - 17 *human-written* pilot questions (from `data/annotations/gold/`);
  - 10 *out-of-scope* questions, whose key terms are automatically checked to be absent from the index.
- **Gold labels.** Gold answers are **time spans**; `eval/gold.py` maps them to the current segments (a segment counts if it overlaps the span by at least half of its own length, or covers at least half of the span). That way labels survive re-segmentation.
- **Retrieval metrics:**
  - Recall@5 and Recall@10: the share of gold segments found;
  - MRR: 1 / rank of the first gold segment;
  - nDCG@10: rewards ranking gold segments higher;
  - each with a bootstrap 95% confidence interval.
- **Refusal.** The threshold that maximizes balanced accuracy on *dev*, reported on *held-out* questions.
- **Generation** (needs an API key):
  - citation accuracy;
  - faithfulness via an LLM-as-judge that checks each claim against only the cited segments;
  - end-to-end refusal;
  - p50/p95 latency per stage.
- **Research harness** (`scripts/run_evaluation.py`, `run_ablations.py`). Segmenters × retrievers, Pk / WindowDiff / boundary F1 for segmentation, paired bootstrap tests, plus the signal-removal, encoder, fusion, segment-length, length-matched and seed ablations.

---

# Part III: The A–Z of ConceptBound

| | Term | What it is here | Why it matters |
|---|---|---|---|
| **A** | **Ablation** | Remove or swap one component and re-measure (signals, encoder, fusion mode, reranker, segment length). | It's the only way to know which part actually helps. |
| **B** | **BM25** | Lexical ranking: term frequency × IDF, length-normalized. | A hard-to-beat baseline. Here it beats dense retrieval (MRR 0.613 vs 0.549). |
| **C** | **Citation validation** | Drop cited labels that weren't retrieved and the sentences relying on them; refuse if none are left. | Every surviving claim points at a real, checkable lecture moment. |
| **D** | **Dense retrieval** | Sentence-BERT embeddings + cosine similarity. | Catches paraphrases that BM25 misses. |
| **E** | **EasyOCR + screen detection** | OCR inside the auto-detected projector screen (a median frame removes the instructor). | Cropping removes the clock overlay that faked "slide changes". |
| **F** | **FAISS `IndexFlatIP`** | Exact inner-product search over L2-normalized vectors (= cosine). | Sub-millisecond at hundreds of segments; approximate indexes only pay off at ~10⁵ vectors or more. |
| **G** | **Gold labels** | Human annotations: answer spans, boundaries, lecture span. | Evaluation is only honest against human ground truth, never pseudo-labels. |
| **H** | **Hybrid retrieval** | 0.5 · max-normalized BM25 + 0.5 · cosine. | The best first-stage system here (MRR 0.633). |
| **I** | **InfoNCE** | Contrastive loss: pick the positive among negatives via softmax. | Trains both Stage 1 (segment InfoNCE) and Stage 2 (query ↔ segment). |
| **J** | **JSON structured output** | The LLM returns `{answer, citations, covered}`, validated by Pydantic. | Machine-checkable answers; malformed output is handled, not crashed on. |
| **K** | **K negatives** | Stage 1 samples K = 4 cross-segment negatives per anchor. | More negatives make a harder, more informative contrastive task. |
| **L** | **Lecture span** | Annotated start and end of actual teaching. | Excludes dead air and the roll call (privacy) from the index and from evaluation. |
| **M** | **Margin loss** | Push boundary-crossing neighbours apart until cos < 1 − m (m = 0.5). | Stops pushing once they're "different enough", which saves capacity. |
| **N** | **nDCG@10** | Discounted gain of gold segments, normalized by the ideal ranking. | Rewards putting the right moment *higher*, not just somewhere in the top 10. |
| **O** | **Out-of-scope questions** | 10 questions whose topics never appear in the lectures (checked automatically). | They measure whether the system refuses instead of hallucinating. |
| **P** | **Pseudo-labels / pseudo-queries** | Automatic boundaries and queries derived from the data itself. | Useful for training; **never** for evaluation (they leak the answer). |
| **Q** | **Query projector** | An MLP mapping the query embedding into the fusion space. | Queries and segments must live in the same space to be compared. |
| **R** | **Reranker (cross-encoder)** | `ms-marco-MiniLM-L-6-v2` re-scores the top 20 jointly with the question. | Precise ordering. Input order fixed a regression (MRR 0.556 → 0.666). |
| **S** | **Segments (concept-coherent)** | Stage 1 output, 60–600 s long, one explanation each. | The unit of retrieval and citation; better units mean better answers. |
| **T** | **Topic drift** | 1 − cosine between Sentence-BERT embeddings of consecutive 30-s transcript windows. | A boundary signal from speech. |
| **U** | **Unimodal experts** | Per-modality networks in the pairwise (Arora-style) fusion. | Each modality is processed on its own before pairwise fusion. |
| **V** | **Variance trigger** | The `auto` mode falls back to hybrid when fusion scores look degenerate. | Keeps the app working even with an untrained or collapsed model. |
| **W** | **Whisper** | Speech-to-text, forced to English, with hallucination filters. | Transcripts carry most of a lecture's content. |
| **X** | **Cross-attention fusion** | Self-attention across the `[OCR][ASR][VIS]` tokens. | Learns per segment which modality to trust. |
| **Y** | **Yield of the gate (refusal)** | The confidence gate refuses before the LLM is called; the threshold is calibrated on dev questions. | Prevents confident nonsense on off-topic questions. |
| **Z** | **Z-score normalization** | Each modality's projection is normalized before fusion. | Stops one modality (bigger magnitude) from dominating attention. |

---

# Part IV: How it was built, day by day

Each "Day" is one focused working session. Dates come from the git history.

## Phase 1: Prototype (13 June – 1 October 2026)

### Day 0: Environment and scaffold
- **Built:** a virtual environment; dependencies (PyTorch, transformers, sentence-transformers, Whisper, EasyOCR, OpenCV, FAISS, FastAPI); a folder skeleton (`src/`, `data/`, `checkpoints/`, `app/`, `scripts/`); a central `config.py`; and a `.gitignore` for videos, data and checkpoints.
- **Why:** one source of truth for paths and hyperparameters avoids "works on my machine" drift across a dozen scripts.

### Day 1: Problem scoping and literature
- **Problem statement:** *given a long lecture, find the moment that answers a free-text student question.*
- **Literature read:** topic segmentation (TextTiling), video–language retrieval (CLIP-based retrievers), lecture indexing (OCR + ASR search).
- **Decision:** a **two-stage design**. Stage 1 decides *where* concepts begin and end; Stage 2 decides *what* each segment is about, so questions can be matched.
- **Base paper:** Arora et al., *Text-to-Multimodal Retrieval with Bimodal Input Fusion in Shared Cross-Modal Transformer* (LREC-COLING 2024). It motivates fusing several modalities for text-to-video retrieval.

### Day 2: Feature extraction
- **Built:** frame sampling every 3 s with OpenCV; Whisper transcription; topic drift from Sentence-BERT on rolling windows; EasyOCR on frames with a text-change signal; optical flow; CLIP embeddings. Everything is saved per video as `.npz`.
- **Why 3 s:** fine enough to place boundaries within a few seconds, coarse enough to keep a one-hour lecture at about 1,300 steps (cheap to model).

### Day 3: Stage 1 model and losses
- **Built:** CNN, Transformer and BiLSTM encoders behind one wrapper; the adjacent margin loss and segment InfoNCE; peak-picking pseudo-labels; a padded dataset with a padding mask.
- **Why embeddings instead of a 0/1 boundary classifier:** boundaries are rare (heavy class imbalance), and a similarity-based score is smooth and comparable across lectures.

### Day 4: Stage 1 training and segmentation
- **Built:** a training loop with a video-level train/val split; boundary scores; peak picking with a minimum length; `segment_videos.py`; fixed-window and shot-detection baselines.

### Day 5: Stage 2 design
- **Built:** a fusion encoder treating the three modality embeddings as three tokens with self-attention; z-score normalization; modality-type embeddings; a concat variant for ablation; the query projector.
- **Why attention over only 3 tokens:** it's nearly free (a 3×3 attention matrix) and lets the model weight modalities per segment.

### Day 6: Segment features and Stage 2 training
- **Built:** per-segment aggregation of slide text, transcript and CLIP; pseudo-query generation; symmetric InfoNCE training.

### Day 7: FAISS and the retrieval engine
- **Built:** a FAISS `IndexFlatIP`; the `LectureRetriever` with a BM25 index and a Sentence-BERT fallback.
- **Observed:** the InfoNCE model **collapsed** early with too little training data.
- **Response:** the **variance trigger** that falls back to hybrid text search.

### Day 8: API and frontend
- **Built:** FastAPI `GET /api/search`; a single-page HTML frontend with "jump to moment" links; raw videos served under `/videos`.

### Day 9: Evaluation framework
- **Built:** boundary F1 and IoU; Recall@k; mean IoU@1; an evaluation script against baselines.

### Day 10: Ablation hooks and polish (commits of 13 July, 1 September and 1 October)
- **Built:**
  - CLAHE contrast enhancement for OCR;
  - vectorized text search (latency);
  - a training-loss plot;
  - variance-based collapse detection;
  - deprecation fixes;
  - a concept guide;
  - **support for the college lecture videos** (11 recordings).

## Phase 2: Audit and making it research-grade (15 October 2026)

### Day 11: Independent audit
- The audit asked whether the project could support a research paper. **Verdict: not yet.** Findings:
  1. **OCR was dead:** text was empty and the signal zero in *all 11* lectures.
  2. **Stage 1 was trained and judged on pseudo-labels** derived from its own inputs, and no human boundaries existed (circular).
  3. **Stage 2 was trained and evaluated on pseudo-queries** (the first 12 words of each segment's own transcript, often filler like "I mean, I mean…").
  4. **Leakage:** the train/val split was random over segments of the same lectures.
  5. **The fallback silently replaced the model's results with BM25 ones**, hiding failures.
  6. **Segmentation was degenerate:** 30-second median segments next to 27-minute ones.

### Day 12: Root-causing the dead OCR
- **Cause:** EasyOCR's model download stalled, and `get_ocr_backend` *caught the exception and silently returned a dummy backend that returns empty text*.
- **Fix:** download the weights; make backend loading **raise**; add **screen detection** (median frame → Otsu → largest screen-shaped blob, per 5-minute chunk) so the burnt-in clock isn't read; record provenance (engine, crop source, coverage).
- **Result:** slide text found on **69–93%** of sampled frames (lectures 1–5).

### Day 13: More silent bugs
- **Transcript duplication.** Per-step transcripts were 30-s rolling windows, so joining them repeated every word about 10×. Combined with Sentence-BERT's 256-token limit, a segment's embedding only "saw" its first ~8 seconds. **Fix:** keep raw Whisper segments and build segment text from them; encode long texts by chunking and averaging.
- **Index truncation.** Index metadata kept only the first 300 characters of each segment, and BM25 ran on that. **Fix:** store full text, with a separate display snippet.
- **Random fallback.** `SentenceEncoder.encode` returned **random vectors** if the model failed to load. **Fix:** raise instead.
- **Whisper hallucinations** ("I'm going to go to the next room" over pre-class noise). **Fix:** confidence filters, `condition_on_previous_text=False`, and a repeated-phrase filter.

### Day 14: Honest experimental design
- **Splits.** Video-level splits (`data/splits.json`), so test lectures are never trained on.
- **Pseudo-data separation.** Pseudo-queries moved to `*_pseudo_queries.json` (training only). Better pseudo-queries (most informative sentence, keywords, slide title; filler removed; duplicates dropped); in-batch false negatives masked.
- **Stage 1 labels.** Label sources `gold`/`pseudo`/`auto`; checkpoint selection by gold boundary F1; segment length bounds of 60–600 s with non-maximum suppression.
- **Evaluation harness.** Every segmenter × every retriever (BM25, MiniLM, BGE, RRF hybrid, cross-encoder rerank, fusion); Pk, WindowDiff and boundary F1; bootstrap confidence intervals; paired tests.
- **Annotation tooling.** Transcript sheets, gold templates, a validator with inter-annotator agreement, and annotation guidelines.

### Day 15: Verification
- An **end-to-end smoke test** on a scratch copy with synthetic gold files: splits → refuse to evaluate without gold → train → segment → index → evaluate with every baseline.
- The first full re-extraction run started in the background.

## Phase 3: Paper and ablations (16 October – 4 November 2026)

### Day 16: Choosing a venue
- Researched conference deadlines and scope; target: **IEEE ICALT 2027** (track AISLE, 5-page limit, deadline 15 January 2027).

### Day 17: Paper draft
- **Built:** a Word document generator reproducing the IEEE conference layout; figures (pipeline diagram; real signals of one lecture with annotated answer spans); 22 references, all checked against publisher or Crossref records; a guide version and an anonymized version.
- **Rule kept:** results stay as yellow placeholders until measured.

### Day 18: Pilot annotation
- **3 lectures and 17 human-written queries** (median answer span 123 s), loaded as gold files.
- **Pilot split:** the annotated lectures are the test set.

### Day 19: Base-paper fusion and ablations
- **Built:** the `pairwise` fusion mode (unimodal experts, a shared pairwise fusion block, per-pair losses), `--drop-signals`, and `run_ablations.py` (signals, encoder, fusion, segment length, length-matched control, seeds).

### Day 20: Measurement tools
- **Built:** `measure_runtime.py` (hardware, extraction time, query latency); `asr_wer.py` (Whisper word error rate against typed reference clips); `collect_paper_results.py` (every paper number in one JSON); a data-driven paper generator; and `run_paper_pipeline.py` (everything after extraction, in one command).

## Phase 4: The RAG layer (4 – 6 November 2026)

### Day 21: Audit and plan
- **Findings:** the stale index was built from broken features; the retriever re-encoded every segment at startup (~30 s); there was no LLM key on the development machine.
- **Plan:** reuse the existing retrieval, and add `src/retrieval`, `src/rag`, `src/llm`, `eval/` and `space/`.

### Day 22: Offline index build
- **Built:** `python -m src.rag.build_index`: segments, embeddings, FAISS, metadata and cached Sentence-BERT embeddings. The app now starts in about 1.5 s.

### Day 23: RAG pipeline and LLM providers
- **Built:** the reranker; the confidence score; the context builder; the prompt; Pydantic parsing; citation validation (including removing sentences that only had fake citations); follow-up rewriting; Groq, Gemini, Ollama and mock clients with backoff.
- **Surprise:** "What is the capital of France?" was retrieved with high confidence. It turned out a Lecture 3 slide really shows that exact example prompt. Lesson: out-of-scope test questions must be *checked* against the transcripts, so the question script now does it automatically.

### Day 24: The Welsh bug
- **Found:** while drafting evaluation questions, Lecture 3's transcript turned out to be **Welsh gibberish**. Lectures 3, 4 and 6 had been auto-detected as Welsh (only 23–29% English-looking segments).
- **Fix:** force English; add a language sanity check; re-transcribe. Now **95% / 93%** English-looking for lectures 3 and 4. (Story 1.)

### Day 25: Evaluation
- **Built:** a 76-question set; `python -m eval.run`.
- **First result:** hybrid is best; **the reranker made things worse.**
- **Diagnosis:** the input order put noisy OCR text first. Transcript-first gave MRR **0.556 → 0.666**. Both rows are kept in the report, with a note that the fix was found on the same questions. (Story 2.)

### Day 26: API, demo, tests and docs
- **Built:** `POST /ask` and `GET /health`; the Gradio demo with embedded players at the timestamp; the Hugging Face Space packager with a privacy guard; **25 pytest tests** with a mocked LLM; README, DEVLOG and a manual task list.

---

# Part V: Results

All from `reports/eval_results.md`: 157 segments, 5 lectures, 66 in-scope questions (49 drafted, **not yet human-verified**, plus 17 human-written) and 10 out-of-scope. CPU only.

### Retrieval (all 66 in-scope questions)

| System | Recall@5 | Recall@10 | MRR [95% CI] | nDCG@10 | p50 latency |
|---|---|---|---|---|---|
| BM25 only | 0.637 | 0.743 | 0.613 [0.52, 0.70] | 0.588 | 8 ms |
| Dense (Sentence-BERT) only | 0.576 | 0.714 | 0.549 [0.45, 0.65] | 0.544 | 12 ms |
| Dense (fusion encoder) only | 0.221 | 0.379 | 0.202 [0.13, 0.28] | 0.226 | 12 ms |
| **Hybrid (BM25 + Sentence-BERT)** | **0.673** | 0.764 | 0.633 [0.54, 0.72] | 0.611 | 12 ms |
| Current app mode (`auto`) | 0.673 | 0.764 | 0.633 [0.54, 0.72] | 0.611 | 19 ms |
| Hybrid + reranker, OCR-first input (original) | 0.524 | 0.686 | 0.556 [0.46, 0.66] | 0.535 | 2.5 s |
| Hybrid + reranker, transcript-first input | 0.642 | **0.777** | **0.666** [0.58, 0.76] | **0.623** | 1.8 s |

### Human-written questions only (17)

| System | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 only | 0.582 | 0.596 | 0.583 |
| Hybrid | 0.597 | **0.630** | **0.590** |
| Hybrid + reranker (transcript-first) | 0.460 | 0.521 | 0.527 |

### Refusal gate (threshold from 30 dev questions, reported on 29 held-out)

| Confidence source | Threshold | Balanced accuracy | Out-of-scope refused | In-scope wrongly refused |
|---|---|---|---|---|
| Dense similarity | 0.320 | 0.875 | 0.75 | 0.00 |
| Reranker probability | 0.001 | 1.000 | 1.00 | 0.00 |

### Generation
**Not run yet**, because no LLM API key was available on the development machine. Faithfulness, citation accuracy, end-to-end refusal and LLM latency appear in the report only after `python -m eval.run` runs with a key.

### Reading the results
1. **Hybrid beats either retriever alone.** BM25 and dense fail on different questions.
2. **BM25 beats dense here.** Lecture questions are full of exact technical terms ("perplexity", "bigram", "LSTM").
3. **The reranker is a mixed result.** Fixing its input made it best on the full set, but on the 17 human-written questions it is worse than hybrid and adds about 1.8 s. It stays switchable until more human questions decide.
4. **The fusion encoder is the weakest part** (MRR 0.20). It needs retraining on real questions.
5. **The refusal gate works on these questions**, but the held-out set has only a handful of out-of-scope questions, so it's a weak estimate.

---

# Part VI: Lessons, limitations and next steps

### Lessons
1. **Silent failures are the most dangerous bugs.** Empty OCR, random-vector fallbacks, repeated transcripts and Welsh decoding all *ran without errors*. Validate data, not just code, and make pipelines fail loudly.
2. **Baselines first.** BM25 is hard to beat; every learned component must earn its place in an ablation.
3. **Don't evaluate on what you trained on.** Pseudo-queries come from the segment's own text, so finding them is trivial.
4. **Report the bad numbers too.** The original reranker row and the weak fusion encoder stay in the tables.
5. **Input construction is part of the model.** Field order and truncation changed reranker MRR by 0.11.

### Limitations
- One institution, one course, one camera; 5 lectures indexed; small evaluation set with wide confidence intervals.
- 49 of the 66 in-scope questions are drafted from transcripts and not yet human-verified (they favour lexical retrieval).
- No human boundary annotations yet, so no segmentation scores.
- The fusion encoder was trained on pseudo-queries.
- No generation metrics yet (no API key).
- LLM-as-judge limitations: leniency towards fluent text, shared biases with the generator, inconsistent claim splitting.
- Privacy: recordings include students, so no public release without consent.

### Next steps
1. Verify the drafted questions; annotate boundaries; add a second annotator for agreement.
2. Add an LLM key and report generation metrics.
3. Retrain Stage 2 on human questions; re-run the fusion ablation (cross-attention vs concat vs pairwise).
4. Index all lectures (about 30 planned); build a public demo on openly licensed lectures.
5. A small user study: how long does a student need to find an answer with and without ConceptBound?

---

# Part VII: Interview questions and answers

### Project overview
1. **Explain the project in 30 seconds.** ConceptBound lets students ask questions about recorded lectures. It cuts each lecture into concept segments with a trained boundary detector, indexes them with transcript, slide text and visual features, retrieves the right moments with hybrid BM25 plus dense search, and has an LLM answer **only from those moments**, with clickable `[Lecture, mm:ss]` citations. It refuses when the lectures don't cover the question.
2. **What was your role / the hardest part?** Building the end-to-end pipeline and, hardest of all, finding the silent data failures (empty OCR, Welsh transcripts, repeated text) that made the numbers wrong without any error message.
3. **Why two stages?** Stage 1 decides *where* concepts start and end; Stage 2 decides *what* each segment is about. Concept-sized units give better retrieval targets and cleaner LLM context than arbitrary windows.

### Data and features
4. **Why extract features every 3 seconds?** It's fine enough to place boundaries within seconds, and small enough (about 1,300 steps per hour) for cheap sequence modelling.
5. **How did you get clean slide text from a wide classroom shot?** I detected the projector screen automatically: the median frame removes the moving instructor, Otsu thresholding finds the bright area, and I keep the largest screen-shaped region per 5-minute chunk. Then I OCR only that crop, and only when the screen changes (SSIM).
6. **What is topic drift?** The cosine distance between Sentence-BERT embeddings of consecutive 30-second transcript windows. It's high when the spoken topic changes.
7. **How do you handle Whisper hallucinations?** Confidence filters (`no_speech_prob`, `avg_logprob`, compression ratio), no conditioning on previous text, and removal of short phrases repeated in 60 s.

### Models
8. **How are positive and negative pairs formed in Stage 1?** Same-segment neighbours are positives and boundary-crossing neighbours are negatives (margin loss). For InfoNCE, each step is an anchor with one same-segment positive and four other-segment negatives.
9. **What does the margin do?** It only penalizes boundary-crossing neighbours while their similarity is above 0.5, so the model doesn't waste capacity pushing them further apart.
10. **Why an embedding-based boundary score rather than a classifier?** Boundaries are rare (class imbalance); a similarity score is smooth, comparable across lectures and naturally peak-pickable.
11. **Explain the Stage 2 fusion encoder.** Three modality embeddings become three tokens with type embeddings; one self-attention block lets them attend to each other; pooled and normalized into one segment vector. A query projector maps questions into the same space.
12. **What are in-batch negatives?** In a batch of B pairs, each query's other B−1 segments act as negatives, so no extra negative mining is needed.
13. **Why mask same-segment pairs in the loss?** Two questions can target the same segment; treating it as a negative would push identical items apart (a false negative).
14. **What's the pairwise fusion variant?** It's the base paper's idea: unimodal experts, one shared block fusing each modality pair, plus a loss per pair. It's an ablation against the default.

### Retrieval
15. **Why FAISS `IndexFlatIP` and not HNSW or IVF?** It's exact, and inner product equals cosine on normalized vectors. With hundreds of segments it's sub-millisecond; approximate indexes add recall loss and tuning for no gain at this scale.
16. **BM25 vs dense retrieval?** BM25 is exact-term matching: precise on jargon, untrained, explainable, but blind to synonyms. Dense matches meaning but can drift and is weaker on rare tokens. Here BM25 actually wins (0.613 vs 0.549 MRR).
17. **How is hybrid computed?** Top candidates from both; BM25 divided by its max to fall in 0–1; averaged 50/50 with cosine. Reciprocal rank fusion is a scale-free alternative, used in the research harness.
18. **Why does a reranker help in general?** Bi-encoders embed the query and passage separately; a cross-encoder reads them together, so it matches specific words across both. It's too slow for the whole corpus, so it only re-scores the top 20.
19. **Did the reranker help here?** It depends on the input: OCR-first made it worse (MRR 0.556); transcript-first made it best overall (0.666). But on human-written questions it's below hybrid, so it's switchable, not trusted blindly.

### RAG and LLM
20. **How does RAG reduce hallucination?** The LLM answers from retrieved evidence, is told to say "not covered" otherwise, a confidence gate blocks off-topic questions before the LLM, and citations are validated. It reduces hallucination; it doesn't make it impossible.
21. **What is the refusal gate and how is its threshold chosen?** If retrieval confidence (reranker probability or best cosine) is below the threshold, the system refuses without calling the LLM. The threshold maximizes balanced accuracy on dev questions and is reported on held-out ones.
22. **What happens if the LLM cites a segment that wasn't retrieved?** The label is dropped, sentences that relied only on it are removed, and if no valid citation remains the answer becomes a refusal.
23. **Why `[S1]`-style labels instead of real segment IDs in the prompt?** LLMs copy short labels reliably; the code maps them back to the exact IDs.
24. **How do you handle malformed JSON from the LLM?** A tolerant parser (strip fences, find the first JSON object, coerce types) plus Pydantic validation. On failure the API returns the top moments with `mode="llm_error"`; it never crashes.
25. **How are follow-up questions handled?** Short or pronoun-heavy questions get the previous two user questions prepended before retrieval. It's deterministic and testable.
26. **How do you manage the token budget?** At most 1,800 tokens of context, about 450 per segment, estimated as words × 1.3; overlapping segments are removed first.
27. **How do you switch providers and handle rate limits?** An `LLMClient` interface chosen by `LLM_PROVIDER`; retries with exponential backoff and jitter, `Retry-After` support, and clear error types; the app degrades to retrieval-only mode.

### Evaluation
28. **How did you evaluate retrieval?** Recall@5/10, MRR and nDCG@10 against gold segments, with bootstrap confidence intervals, as an ablation of BM25, dense, fusion, hybrid, auto and reranked hybrid. Human-written questions are reported separately.
29. **Why store gold answers as time spans, not segment IDs?** IDs change every time lectures are re-segmented; time spans don't. The evaluation maps spans to the current segments.
30. **Why are pseudo-queries never used for evaluation?** They're built from the segment's own text, so lexical search finds them trivially and the score means nothing.
31. **What is faithfulness, and what are the limits of LLM-as-judge?** The share of answer claims supported by the cited segments, judged by an LLM with a fixed rubric. Limits: leniency to fluent text, shared biases with the generator, inconsistent claim splitting, and cost. Spot-check by hand.
32. **How do you know the out-of-scope questions are really out of scope?** The question script searches the index for each topic's key terms and warns if any appear. That caught "capital of France", which a slide really contains.

### Engineering
33. **Why build the index offline?** Whisper and OCR take minutes per lecture on CPU; the app loads precomputed files and starts in about 1.5 s on a free CPU.
34. **How did you test an LLM app without paying for API calls?** A mock `LLMClient` with scripted outputs; 25 pytest tests cover citation validation, refusal, malformed output, retries and the factory.
35. **How did you handle privacy?** Transcripts and indexes are git-ignored; `lecture_span` removes the roll call; the Space packager refuses internal lectures; the public demo must use openly licensed videos.
36. **How would you scale to 10,000 lectures?** An approximate FAISS index (HNSW or IVF-PQ), a search engine for BM25, batch or GPU extraction, a smaller or GPU reranker, caching, async LLM calls, and a larger human-labelled evaluation set.
37. **What would you do with one more month?** A verified 100+ question evaluation set, boundary annotations, retraining Stage 2 on human questions, generation metrics, and a user study on time-to-answer.

### Behavioural
38. **Tell me about a time your results were wrong.** See Story 1 (silent data failures).
39. **Tell me about a time an "improvement" made things worse.** See Story 2 (the reranker).
40. **How do you make sure you're not fooling yourself?** Human gold data only, baselines in every table, dev/test separation for thresholds, confidence intervals, and reporting the bad numbers as well.

---

# Part VIII: Two interview stories

## Story 1: "Tell me about a difficult bug you found and how you solved it."

**Situation.** My lecture search system was producing numbers and running without a single error. While preparing evaluation questions I read the transcript of Lecture 3 and found it was gibberish: *"Yn yw'n gweithio'r ddod…"*, which is Welsh, for an English lecture.

**Task.** I needed to find out how far the corruption went, fix it at the root, and make sure the same class of bug couldn't hide again, because every downstream number (topic drift, retrieval, the evaluation) depended on those transcripts.

**Action.**
1. **Measured the damage.** I wrote a quick check counting how many transcript segments contain common English function words. Lectures 3, 4 and 6 scored only 23–29%; the others were around 90%.
2. **Found the root cause.** Whisper auto-detects the language from the **first 30 seconds** of audio, and a classroom recording starts with room noise and chatter. So it guessed Welsh and decoded the whole hour as Welsh.
3. **Fixed it at the source.** I forced `language="en"` in the configuration and re-transcribed only the three affected lectures.
4. **Prevented recurrence.** I added a language sanity check that prints a loud warning when less than half of a transcript looks like English.
5. **Applied the same lens to the rest of the pipeline**, looking for anything that "succeeds" but produces garbage. That surfaced earlier silent failures too: OCR had been falling back to an empty backend for *every* lecture, and transcripts were being repeated about ten times so the embeddings saw only a segment's first seconds. I made those paths raise errors instead of silently continuing.

**Result.** English-looking transcript share went from 25% to 95% for Lecture 3, and from 29% to 93% for Lecture 4. I rebuilt the index and re-ran the evaluation on clean data. The biggest lesson: *validate the data, not just the code*. A pipeline that never crashes can still be completely wrong, so I now build in sanity checks that fail loudly.

## Story 2: "Tell me about a time an improvement made things worse."

**Situation.** I added a cross-encoder reranker on top of my hybrid retriever. That's the textbook way to improve ranking quality, and I expected a clear gain. When I ran the evaluation on 66 questions, the reranker made results **worse**: MRR dropped from 0.633 (hybrid) to 0.556, and it added about 2 seconds per query.

**Task.** Decide whether the technique was wrong for this domain or whether I had built it wrong, without fooling myself into a false win.

**Action.**
1. **Thought about what the model actually sees.** A cross-encoder reads at most **512 tokens**. My code fed it the segment's **OCR text first**, then the transcript, cut at 1,500 characters. The OCR text from a classroom camera is noisy, so the model spent most of its window on OCR garbage and the first few sentences of a several-minute explanation.
2. **Ran a controlled diagnostic** on the same questions with three input variants: OCR-first (original), transcript-first, and MaxP (score 150-word passages and take the best).
3. **Results:** transcript-first lifted MRR to **0.666** and Recall@10 to **0.777**, the best of all systems. MaxP didn't help (0.559).
4. **Reported it honestly.** I made transcript-first the default but **kept the original row** in the results table, with a note that the fix was found on the same question set, so its gain is optimistic.
5. **Checked a harder subset.** On the 17 human-written questions the reranker was still below plain hybrid (MRR 0.521 vs 0.630).

**Result.** The reranker went from the worst to the best system on the full set. Because of the human-question result, I kept it **switchable** instead of declaring victory, and noted that more human-written questions are needed to decide. The lesson: *input construction is part of the model*, and an "improvement" needs the same scrutiny as a regression.

---

# Appendix: command cheat sheet and file map

```bash
# Demo
python space/app.py                                   # Gradio on http://127.0.0.1:7860
uvicorn app.backend.main:app --port 8000              # API, docs at /docs

# Build / rebuild the RAG index (no Whisper/OCR unless --extract-missing)
python -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05

# Evaluate the RAG layer
python -m eval.make_questions && python -m eval.run   # -> reports/eval_results.md

# Tests
python -m pytest tests -q

# Research pipeline
python scripts/extract_features.py                    # Whisper + OCR + CLIP (slow)
python scripts/refresh_features.py --ocr --asr        # re-run OCR/ASR on existing features
python scripts/run_paper_pipeline.py                  # train, segment, evaluate, ablate, rebuild paper
```

| Path | Purpose |
|---|---|
| `src/feature_extraction/` | Whisper, OCR with screen detection, CLIP, optical flow |
| `src/stage1_boundary/` | Boundary detector: model, losses, pseudo-labels, training, segmentation |
| `src/stage2_retrieval/` | Fusion encoder, training, FAISS index, original hybrid retriever |
| `src/retrieval/` | Search service, reranker, confidence |
| `src/rag/` | Index build, context, prompts, schema, citations, rewrite, pipeline |
| `src/llm/` | Provider interface and clients |
| `eval/` | RAG evaluation (questions, metrics, judge, runner) |
| `scripts/` | Research pipeline, annotation tools, ablations, measurement |
| `app/` | FastAPI backend and HTML frontend |
| `space/` | Gradio demo and Hugging Face packaging |
| `tests/` | pytest suite |
| `paper/` | ICALT 2027 paper draft generator and figures |
| `docs/ANNOTATION_GUIDELINES.md` | How to annotate boundaries and queries |
| `MANUAL_TASKS.md` | What still needs a human |
