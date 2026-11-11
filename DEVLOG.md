# DEVLOG: ConceptBound

> The complete story of this project. It covers:
> - what it does, and how every part works and *why* it was built that way;
> - the important source code;
> - how to run it and how it is deployed;
> - how it was built, day by day;
> - what the measurements show;
> - what went wrong and how it was fixed;
> - how to talk about it in an interview.
>
> **Rules for this file.** Every number comes from a file in the repo (mainly `reports/eval_results.md`); if a number isn't in a report, it isn't quoted here. Code excerpts are copied from the real source files and trimmed to the important lines. The file path above each excerpt says where to read the full version.

---

## Contents

- [Part I: The problem and the idea](#part-i-the-problem-and-the-idea)
- [Part II: Running the project](#part-ii-running-the-project)
- [Part III: How it works, end to end (with code)](#part-iii-how-it-works-end-to-end-with-code)
  - [3.1 The data](#31-the-data)
  - [3.2 Feature extraction](#32-feature-extraction-srcfeature_extraction)
  - [3.3 Stage 1: Pedagogical Boundary Detector](#33-stage-1-pedagogical-boundary-detector-srcstage1_boundary)
  - [3.4 Stage 2: Cross-Modal Segment Fusion Encoder](#34-stage-2-cross-modal-segment-fusion-encoder-srcstage2_retrieval)
  - [3.5 Indexing](#35-indexing-srcragbuild_indexpy)
  - [3.6 Retrieval](#36-retrieval)
  - [3.7 The RAG pipeline](#37-the-rag-pipeline-srcrag)
  - [3.8 LLM providers](#38-llm-providers-srcllm)
  - [3.9 Serving: API and demo](#39-serving-api-and-demo)
  - [3.10 Evaluation](#310-evaluation-eval)
- [Part IV: Deployment](#part-iv-deployment)
- [Part V: The A–Z of ConceptBound](#part-v-the-az-of-conceptbound)
- [Part VI: How it was built, day by day](#part-vi-how-it-was-built-day-by-day)
- [Part VII: Results](#part-vii-results)
- [Part VIII: Lessons, limitations and next steps](#part-viii-lessons-limitations-and-next-steps)
- [Part IX: Interview questions and answers](#part-ix-interview-questions-and-answers)
- [Part X: Two interview stories](#part-x-two-interview-stories)
- [Appendix: command cheat sheet and file map](#appendix-command-cheat-sheet-and-file-map)

---

# Part I: The problem and the idea

## 1.1 The problem

Our department records every lecture with a fixed auditorium camera. Students use the recordings to revise, but a one-hour video is a poor revision tool. A student who wants "the part where the professor explains why we use softmax" has to scrub through the timeline, guess, and scrub again. Multiply that by 11 recordings (about 30 by the end of the semester) and finding one explanation can take longer than re-reading the textbook.

Ordinary keyword search doesn't solve it either:

- **Student wording differs from lecture wording.** "Why can't an RNN remember far-back words?" is the same question as "long-term dependencies", but shares almost no words with it.
- **The evidence is spread across modalities.** Part of the answer is spoken, part is on the slide, and part is in a diagram.
- **Classroom recordings are messy:**
  - the projector screen covers only part of a wide shot;
  - a clock is burnt into the corner of every frame;
  - there is dead air before class;
  - speech recognition makes mistakes on accented, spontaneous speech.
- **A list of moments isn't an answer.** Even a perfect search engine returns five timestamps; the student still has to watch all five to piece the answer together.

## 1.2 The idea

1. **Cut each lecture into concept-sized pieces** (not fixed 60-second windows), so each piece is one explanation.
2. **Describe each piece with everything we know about it:** what was said, what was on the slide, and what the frame looked like.
3. **Search those pieces with a strong hybrid retriever**, so a student's question finds the right moment in the right lecture, across all lectures at once.
4. **Answer in plain language with a large language model that may only use the retrieved pieces.** Every sentence cites its source as `[Lecture 3, 12:45]`. If the lectures don't cover the question, the system says so instead of guessing.

The name **ConceptBound** comes from step 1: everything in the system is bound to concept-level segments. They are the unit that is searched, the unit the LLM reads, and the unit that is cited.

## 1.3 Design principles

| Principle | What it means in practice |
|---|---|
| **Grounded, not generative** | The LLM only paraphrases retrieved lecture content; every claim needs a valid citation. |
| **Fail loudly** | No silent fallbacks to empty or random data; checks for wrong-language transcripts, empty OCR, and so on. |
| **Measure before claiming** | Every component is compared against simple baselines on human-annotated data; nothing is reported that wasn't measured. |
| **Offline heavy, online light** | Whisper, OCR and CLIP run once while indexing; the app only loads precomputed files and runs on a free CPU. |
| **Explainable choices** | Standard, well-understood techniques (BM25, Sentence-BERT, FAISS, cross-encoders, InfoNCE) over exotic ones. |
| **Degrade, don't crash** | No API key: retrieval-only mode. LLM down or rate-limited: show the moments with a clear message. Bad JSON: same. |
| **Privacy first** | Classroom video shows students; videos, transcripts and indexes stay out of git; the deployed demo ships no video. |

## 1.4 What the user sees

1. They type *"Why is softmax used at the output layer?"*.
2. About 3 seconds later they get a short answer, for example: *"Softmax turns the network's raw scores into a probability distribution over the classes … [Lecture 1, 48:33] [Lecture 5, 19:03]"*.
3. Under the answer, each citation shows the exact quote from that moment and, in the local version, a video player that starts at that second.
4. They can ask a follow-up such as *"explain it more simply"*, and the system knows what "it" refers to.
5. If they ask something the course never covered (*"How do I configure a VLAN?"*), they get *"This isn't covered in the indexed lectures."*

---

# Part II: Running the project

Everything below is run from the project root (the folder containing `config.py`). The commands are written for Windows PowerShell or Git Bash; on macOS or Linux, replace `.venv\Scripts\` with `.venv/bin/`.

## 2.1 What you need

| Requirement | Why |
|---|---|
| **Python 3.10+** (developed on 3.12) | The whole project is Python. |
| **ffmpeg** on `PATH` | Whisper uses it to read audio. Only needed for feature extraction. |
| **About 16 GB RAM** | Feature extraction holds one lecture's sampled frames in memory. The demo needs far less. |
| **GPU (optional)** | Everything runs on CPU; extraction is just slower (minutes per lecture). |
| **Free Groq API key (optional)** | Written answers. Without it, the demo still shows the most relevant moments. |

## 2.2 Install (once)

```bash
git clone https://github.com/marshallmathers0/Concept-Bound-Capstone.git
cd Concept-Bound-Capstone
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2.3 Add the LLM key (once)

```bash
copy .env.example .env             # macOS/Linux: cp .env.example .env
```

Open `.env` and set:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...your key...    # from https://console.groq.com/keys
# LLM_MODEL=openai/gpt-oss-20b     # optional; this is the default
```

- `.env` is listed in `.gitignore`, so the key is never committed.
- The code reads keys only from environment variables (`.env` locally, Space secrets when deployed).
- `src/llm/factory.py` finds `.env` by its absolute path, so the app works no matter which folder you start it from.

## 2.4 Run the demo (every time)

```bash
.venv\Scripts\python.exe space\app.py
```

1. Wait for `Running on local URL: http://127.0.0.1:7860` (the first start downloads three small models, about 1–2 minutes; later starts take seconds).
2. Open http://127.0.0.1:7860.
3. Click an example question or type one.
4. Optional:
   - tick **Show retrieved segments** to see the ranking and scores;
   - untick **Use cross-encoder reranker** to compare speed and ranking;
   - click **New conversation** to reset follow-up context.
5. Stop it with `Ctrl+C` in the terminal.

The line under each answer shows what happened, for example:

```
LLM: groq (openai/gpt-oss-20b) — confidence 1.00 — rewrite: 0 ms | retrieval: 36 ms | rerank: 1873 ms | llm: 1484 ms | llm_request: 1483 ms | total: 3395 ms
```

`llm_request` is the model's own response time; `llm` also includes any time spent waiting on Groq's rate limit.

## 2.5 Run the API

```bash
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --port 8000
```

- Interactive docs: http://127.0.0.1:8000/docs
- Ask a question:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Why is softmax used at the output layer?\"}"
```

- Health check: `curl http://localhost:8000/health`

## 2.6 Run the tests

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

33 tests, about 30 seconds. They use a mocked LLM, so they need no API key and cost nothing.

## 2.7 Run the evaluation

```bash
.venv\Scripts\python.exe -m eval.make_questions      # builds eval/questions.jsonl (76 questions)
.venv\Scripts\python.exe -m eval.run                 # writes reports/eval_results.md and .json
```

- Without a key, it runs the retrieval ablation and the refusal calibration only (a few minutes).
- With a key, it also runs the generation evaluation. On Groq's free tier it pauses 45 s between questions to stay under the 8,000 tokens-per-minute limit, so it takes about an hour.
- Options: `--no-generation` skips the LLM part; `--llm-pause 10` shortens the pause on a paid tier.

## 2.8 Rebuild everything from raw videos

Only needed for new lectures or after changing the models.

```bash
# 1. Put lecture videos in data/raw_videos/  (e.g. college_lec06.mp4)

# 2. Extract features, train both stages and build the base index (slow: Whisper + OCR + CLIP)
.venv\Scripts\python.exe scripts\run_full_pipeline.py

# 3. Build the RAG index used by the demo (reuses features; no Whisper/OCR)
.venv\Scripts\python.exe -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05
```

Then describe each video in `data/video_meta.json` (title, URL, licence, attribution). Building the RAG index for 5 lectures takes about a minute.

## 2.9 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Groq request failed (HTTP 404) … model_not_found` | Groq retired the configured model. | Already handled: the client switches to an available model automatically. Remove any old `LLM_MODEL` line from `.env`, then **restart the app**, because a running app keeps the old code. |
| "The language model is unavailable … rate limit (HTTP 429)" | More than 8,000 tokens in one minute on Groq's free tier. | Wait a minute. The app retries with backoff and falls back to showing the moments. |
| "No LLM is configured" | No key in `.env`, or `LLM_PROVIDER=none`. | Add the key (section 2.3) and restart. |
| `ConnectionResetError [WinError 10054]` traceback in the terminal | Windows prints it when the browser drops a connection. Harmless. | Already silenced in `space/app.py`. |
| `OSError: [Errno 10048]` or the port is in use | Another copy of the app is running. | Close it, or set `GRADIO_SERVER_PORT=7861`. |
| `RAG pipeline not available` from the API | The RAG index was not built. | Run step 3 of section 2.8. |

---

# Part III: How it works, end to end (with code)

```
OFFLINE (once per lecture)
 video ─┬─► Whisper ASR ──────────────► transcript segments (start, end, text)
        ├─► screen detection + EasyOCR ► slide text per 3 s step
        └─► CLIP + optical flow ───────► visual embedding + visual change per step
                     │
                     ▼
        per-step signal stream (every 3 s, 35 numbers)
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
   ─► citation validation ─► answer with [Lecture, mm:ss] citations + latency
```

**Two ideas to keep in mind while reading:**
1. *Offline* code (Whisper, OCR, CLIP, both models, index build) is slow, runs once, and produces files.
2. *Online* code (retrieval, rerank, LLM) is fast and only reads those files. That's why the deployed demo needs no video at all.

## 3.1 The data

- **11 recordings** of an undergraduate course on large language models, each exactly 64 minutes, 1080×608 at 25 fps, from one fixed auditorium camera. About 30 are planned by the end of the semester.
- **What the camera sees:**
  - the projected slides in the left half of the frame;
  - the stage and the instructor;
  - part of the audience;
  - burnt-in overlays: a clock in the top right and watermarks at the bottom.
- **Quirks that matter:**
  - **Dead air:** pre-class chatter, breaks, and in one file about 37 minutes of empty stage after class.
  - **A roll call:** one lecture includes students' names being read out.
  - **Changing framing:** camera framing changes between sessions.
- **Sampling:** every recording is sampled every **3 seconds**, giving **1,281 steps per lecture** (64 × 60 / 3 = 1,280, plus the first frame).
- **Indexed so far:** 5 lectures (`college_lec01` to `college_lec05`), 157 concept segments.
- **Metadata:** `data/video_meta.json` holds each video's title ("Lecture 1"), URL, licence and attribution. All entries are marked *"internal – not for public release"*.
- **Annotations:** `data/annotations/gold/` holds the human pilot annotations for 3 lectures (17 questions with answer spans) and each lecture's `lecture_span` (when teaching actually starts and ends).

## 3.2 Feature extraction (`src/feature_extraction/`)

### Speech: Whisper (`asr_extractor.py`)

- **Model.** Whisper `base` (`config.MODELS.whisper_model`), which runs on a laptop CPU. It outputs timestamped segments with confidence fields.
- **Forced English.** `language="en"`. *Why:* Whisper guesses the language from the first 30 s of audio, which in a classroom is noise. It decoded three English lectures as Welsh (Story 1 in Part X).
- **No conditioning on previous text.** `condition_on_previous_text=False`. *Why:* with conditioning, one hallucinated sentence becomes the prompt for the next window, which makes it loop.
- **Hallucination filters.** Whisper invents text over silence ("I don't know. I don't know."). We drop segments that are:
  - empty;
  - highly repetitive (gzip compression ratio > 2.4);
  - probably silence (`no_speech_prob` > 0.6 *and* `avg_logprob` < −1.0);
  - short phrases (≤ 6 words) repeated 3 or more times within 60 s.
- **Sanity check.** `english_share()` warns loudly if fewer than 50% of segments look like English.

The core of `transcribe()` and the filters:

```python
# src/feature_extraction/asr_extractor.py (trimmed)
def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
    model = self._load()
    # condition_on_previous_text=False stops one hallucination from seeding the next window
    result = model.transcribe(audio_path, verbose=False, language=MODELS.whisper_language or None,
                              condition_on_previous_text=False)
    kept = [TranscriptSegment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result["segments"] if not is_probable_hallucination(s)]
    kept = drop_repeated_segments(kept)
    if english_share([s.text for s in kept]) < 0.5:
        print("WARNING: only ...% of transcript segments look like English. Check the language setting.")
    return kept


def is_probable_hallucination(seg, no_speech_thresh=0.6, logprob_thresh=-1.0, compression_thresh=2.4) -> bool:
    text = seg.get("text", "").strip()
    if not text:
        return True
    if seg.get("compression_ratio", 0.0) > compression_thresh:          # repetitive text
        return True
    if seg.get("no_speech_prob", 0.0) > no_speech_thresh and seg.get("avg_logprob", 0.0) < logprob_thresh:
        return True                                                     # silence transcribed anyway
    return False


def english_share(texts) -> float:
    """Fraction of segments whose words are at least 15% common English function words."""
    common = set("the a is and to of in that this we you it so what are for be on with have can do not if one will".split())
    n = hits = 0
    for t in texts:
        w = re.findall(r"[a-z]+", t.lower())
        if w:
            n += 1
            hits += sum(x in common for x in w) / len(w) > 0.15
    return hits / n if n else 0.0
```

**Interview angle:** `english_share` is deliberately crude. It doesn't need to be a language detector; it only needs to scream when an English lecture comes out as something else. Cheap checks that fail loudly beat clever checks nobody runs.

**Topic drift (a Stage 1 signal).** For every 3-s step, the transcript inside a 30-second window around it is embedded with Sentence-BERT:

```
topic_drift_t = 1 − cosine(window_t, window_{t−1})
```

A cheap TF-IDF pre-filter (`texttiling_prefilter`) skips the neural encoder on stretches where the words clearly don't change.

**Long texts are chunked, not truncated.** `SentenceEncoder.encode_long()` splits text into 150-word chunks, encodes each and averages them. *Why:* Sentence-BERT reads at most 256 tokens, so a 5-minute transcript fed directly was being represented by its first ~20 seconds (a real bug, Day 13).

### Slides: EasyOCR on the auto-detected screen (`ocr_extractor.py`)

**Screen detection.** The projector screen is the largest bright, roughly rectangular area that stays in place for the whole lecture.

```python
# src/feature_extraction/ocr_extractor.py (trimmed)
def detect_screen_bbox(frames, num_samples=60, min_area_frac=0.03, pad_frac=0.01):
    idx = np.linspace(0, len(frames) - 1, min(num_samples, len(frames))).astype(int)
    gray = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in idx])
    median = np.median(gray, axis=0).astype(np.uint8)          # removes the moving instructor
    h, w = median.shape

    _, mask = cv2.threshold(cv2.GaussianBlur(median, (5, 5), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # bright vs dark, threshold chosen automatically
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area, aspect = bw * bh, bw / max(bh, 1)
        fill = cv2.contourArea(c) / max(area, 1)
        if area < min_area_frac * h * w or not (1.0 <= aspect <= 2.6) or fill < 0.6:
            continue                                               # too small, wrong shape, or too ragged
        if area > best_area:
            best, best_area = (x, y, x + bw, y + bh), area
    ...  # returns the box as fractions of the frame, slightly padded
```

**Why each step:**

| Step | Why |
|---|---|
| Per-pixel **median** of 60 frames | The instructor moves and the screen doesn't, so the median "erases" the person. |
| Gaussian blur + **Otsu** threshold | Otsu picks the bright/dark threshold automatically per video, so no magic number. |
| Morphological **close** | Fills small holes (dark text on the slide) so the screen is one blob. |
| Aspect 1.0–2.6, fill ≥ 0.6, area ≥ 3% | Rejects windows, lights and the burnt-in clock. |
| Per **5-minute chunk** (`resolve_ocr_bboxes`) | Framing can change mid-recording. |
| Manual override in `data/ocr_regions.json` | Escape hatch when detection is wrong. |

**Why crop at all:** without it, OCR reads the burnt-in clock, which changes every second and looks like a "slide change" at every step.

**Speed: the SSIM gate.** OCR is slow, and slides change rarely. Before OCR-ing a frame we compare it to the previous one with SSIM (structural similarity); if SSIM ≥ 0.95 (less than 5% visual change), we reuse the previous text. This skips about 80–90% of frames.

**Signal.** *Slide-text change* = 1 − string similarity between consecutive OCR texts (`text_change_rate`).

**Fail loudly.** If EasyOCR can't load, `get_ocr_backend` raises. It used to silently return a dummy backend that returned empty text for every frame (Day 12).

### Visuals: CLIP + optical flow

- A CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) embedding per sampled frame, projected to **32-d** (`FEATURES.clip_dim`).
- Optical-flow magnitude between consecutive frames gives a *visual-change* signal (someone walks in front of the screen, a video plays, the slide changes).

**Per-step feature vector:** `[slide-text change, topic drift, visual change, CLIP(32)]` = **35 numbers every 3 seconds**. Everything is saved per lecture as `data/features/<video_id>.npz`, together with the raw Whisper segments and the OCR text, so later stages never re-run the slow models.

## 3.3 Stage 1: Pedagogical Boundary Detector (`src/stage1_boundary/`)

**Goal.** Find where one concept ends and the next begins.

**Why not just threshold one signal?**
- Slide changes are reliable but rare: several ideas can share one slide, and one idea can span several slides.
- Topic drift reacts to every digression and joke.
- The visual signal fires when someone walks past.

A model can learn to combine all the signals *in context*: a slide change during a steady explanation means less than a slide change plus a topic shift plus a pause. Paper Fig. 2 shows the raw signals for a real lecture.

### The model (`model.py`)

A temporal encoder maps the 35-d sequence to L2-normalized embeddings `z_t`, one per 3-s step. BiLSTM is the default (`STAGE1.encoder_type = "bilstm"`, hidden 128, embedding 64, 2 layers); 1-D dilated CNN and Transformer variants exist for the encoder ablation.

```python
# src/stage1_boundary/model.py (trimmed)
class BiLSTMBoundaryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=2, dropout=0.1):
        super().__init__()
        # hidden_dim // 2 per direction -> hidden_dim total when concatenated
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, embed_dim)


@staticmethod
def boundary_scores(z: torch.Tensor) -> torch.Tensor:
    """b_t = 1 - cos_sim(z_t, z_{t+1}); z is L2-normalized, so the dot product is the cosine."""
    sim = (z[:, :-1, :] * z[:, 1:, :]).sum(dim=-1)
    return 1.0 - sim
```

**Why bidirectional:** whether step *t* is a boundary depends on what comes *after* it as much as what came before ("…so that's softmax. Now, cross-entropy…").

**Why BiLSTM over a Transformer by default:** a lecture is ~1,300 steps. An LSTM is linear in length and has a strong built-in bias for local order. A Transformer is quadratic and needs more data than 11 lectures to learn positional structure. The encoder ablation measures the difference.

**Why embeddings instead of a 0/1 classifier:** boundaries are rare (a few per hundred steps), so a classifier faces heavy class imbalance. A similarity score is smooth, comparable across lectures and easy to peak-pick.

### The loss: how positive and negative pairs are formed (`losses.py`)

**1. Adjacent margin loss.**
- Neighbours *in the same segment* (positive pairs) are pulled together: `(1 − cos)²`.
- Neighbours *across a boundary* (negative pairs) are pushed apart until cosine falls below `1 − m`, with margin `m = 0.5`: `max(0, cos − (1 − m))²`.

```python
# src/stage1_boundary/losses.py
def adjacent_margin_loss(z, is_boundary, margin, pair_mask=None):
    sim = (z[:, :-1, :] * z[:, 1:, :]).sum(dim=-1)            # (B, T-1) cosine of neighbours
    same_mask = (is_boundary == 0).float()
    diff_mask = (is_boundary == 1).float()
    if pair_mask is not None:                                  # ignore padded positions
        same_mask, diff_mask = same_mask * pair_mask.float(), diff_mask * pair_mask.float()

    pull_loss = same_mask * (1.0 - sim).pow(2)                 # same segment: pull together
    push_loss = diff_mask * F.relu(sim - (1.0 - margin)).pow(2)  # boundary: push apart, but only to the margin
    return (pull_loss.sum() + push_loss.sum()) / (same_mask.sum() + diff_mask.sum() + 1e-8)
```

**2. Segment InfoNCE.** For each step (the anchor), take **1 positive** from the same segment and **K = 4 negatives** from other segments. A softmax over the 5 similarities (temperature 0.1) must pick the positive. This gives a non-local training signal beyond immediate neighbours.

```python
# src/stage1_boundary/losses.py (trimmed)
def segment_infonce_loss(z, seg_ids, num_negatives=4, temperature=0.1, valid_mask=None):
    B, T, E = z.shape
    pos_idx, neg_idx, valid = sample_pairs(seg_ids, num_negatives, valid_mask)
    batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(B, T)

    z_pos = z[batch_idx, pos_idx]                                                   # (B, T, E)
    z_neg = z[batch_idx.unsqueeze(-1).expand(-1, -1, num_negatives), neg_idx]       # (B, T, K, E)

    pos_sim = (z * z_pos).sum(-1) / temperature                                     # (B, T)
    neg_sim = (z.unsqueeze(2) * z_neg).sum(-1) / temperature                        # (B, T, K)
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)                    # positive is class 0
    labels = torch.zeros(B, T, dtype=torch.long, device=z.device)

    loss = F.cross_entropy(logits.reshape(-1, 1 + num_negatives), labels.reshape(-1), reduction="none")
    return (loss.reshape(B, T) * valid.float()).sum() / (valid.float().sum() + 1e-8)


def contrastive_boundary_loss(z, is_boundary, seg_ids, margin=0.5, num_negatives=4,
                              temperature=0.1, infonce_weight=0.5, key_padding_mask=None):
    ...
    return adjacent_margin_loss(...) + infonce_weight * segment_infonce_loss(...)
```

Total: **`L = L_adj + 0.5 · L_NCE`**.

**Details worth knowing:**
- **Padding masks.** Lectures in a batch have different lengths; padded steps are excluded from both terms (`key_padding_mask` → `valid_mask`, `pair_mask`).
- **Invalid anchors.** A one-step segment has no positive, so `sample_pairs` marks it invalid instead of using the anchor itself.
- **Why a margin, not "push to −1":** two different concepts can still be related (softmax and cross-entropy). Once they're "different enough" (cosine < 0.5) the model stops spending capacity on them.

### Labels

- **Gold** human boundary annotations where available; otherwise **pseudo-labels** (peaks of the input signals, `pseudo_labels.py`). The label source is chosen with `--label-source gold|pseudo|auto`.
- **Honesty rule:** a model trained only on pseudo-labels can at best re-learn the heuristic, so the heuristic is always reported as a baseline.
- A gold file with queries but no boundaries is *not* treated as "this lecture has no boundaries" (that was a real bug, fixed on Day 14).

### From scores to segments (`segment.py`)

1. **Adaptive threshold.** Peaks above `mean + 1.0 × std` of the lecture's own scores. *Why:* the absolute scale of `1 − cos` depends on embedding size and how long the model trained, so a fixed threshold doesn't transfer between checkpoints.
2. **Length constraints.** Non-maximum suppression, then splitting, so every segment lasts **60–600 s**. *Why:* the first version produced 12-second "concepts" next to 27-minute ones.

```python
# src/stage1_boundary/segment.py (trimmed)
def enforce_segment_lengths(candidate_idx, boundary_scores, min_steps, max_steps):
    T = len(boundary_scores) + 1
    kept = []
    # 1. Non-maximum suppression: strongest candidates first, keep one only if it is
    #    at least min_steps away from every kept boundary and from both ends.
    for t in sorted(candidate_idx, key=lambda c: -boundary_scores[c]):
        start = t + 1
        if start < min_steps or T - start < min_steps:
            continue
        if all(abs(start - (k + 1)) >= min_steps for k in kept):
            kept.append(t)

    # 2. Split: while a segment is longer than max_steps, cut it at its highest-scoring interior point.
    changed = True
    while changed:
        changed = False
        cuts = [0] + sorted(k + 1 for k in kept) + [T]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= max_steps:
                continue
            lo, hi = a + min_steps - 1, b - min_steps - 1
            if hi >= lo:
                kept.append(lo + int(np.argmax(boundary_scores[lo:hi + 1])))
                changed = True
                break
    return np.array(sorted(kept))
```

**Signal-removal ablation.** `--drop-signals ocr|drift|visual|clip` zeroes input channels during training. The mask is saved in the checkpoint, so inference applies the same mask and the comparison is fair.

## 3.4 Stage 2: Cross-Modal Segment Fusion Encoder (`src/stage2_retrieval/`)

Each segment gets three embeddings (`segment_features.py`):
- **slide text:** Sentence-BERT over the *distinct* slide texts shown in the segment (de-duplicated, so a slide shown for 5 minutes doesn't count 100 times);
- **transcript:** Sentence-BERT over the Whisper segments inside the segment, **chunked and averaged**, never truncated;
- **visual:** the mean CLIP embedding over the segment's steps.

### Three fusion modes (the fusion ablation)

| Mode | How | Why it exists |
|---|---|---|
| `cross_attention` (default) | The three tokens `[OCR][ASR][VIS]` get modality-type embeddings and pass through one transformer block (multi-head self-attention + FFN), then mean-pool → MLP → L2-normalize. | Lets the model decide *per segment* which modality to trust. |
| `concat` | Concatenate the three projections, then MLP. | Baseline: no learned interaction between modalities. |
| `pairwise` | Adapted from Arora et al. (LREC-COLING 2024), the project's base paper. Each modality first goes through its own *unimodal expert*; one **shared** transformer block fuses each modality **pair**; pooled; plus a contrastive loss per pair. | Tests the base paper's design in the lecture domain. |

```python
# src/stage2_retrieval/fusion_model.py (trimmed)
class SegmentFusionEncoder(nn.Module):
    def __init__(self, ocr_dim, transcript_dim, visual_dim, hidden_dim=384, out_dim=384,
                 num_heads=4, dropout=0.1, fusion_mode="cross_attention"):
        super().__init__()
        self.ocr_proj = nn.Linear(ocr_dim, hidden_dim)
        self.transcript_proj = nn.Linear(transcript_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.zscore_norm = ZScoreNorm()                                   # equalize modality magnitudes
        self.modality_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)  # "which modality am I?"
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def _fuse_tokens(self, tokens):                       # one post-norm transformer block
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.mean(dim=1)

    def forward(self, ocr_emb, transcript_emb, visual_emb):
        ocr_h = self.zscore_norm(self.ocr_proj(ocr_emb)) + self.modality_embed[0]
        trans_h = self.zscore_norm(self.transcript_proj(transcript_emb)) + self.modality_embed[1]
        vis_h = self.zscore_norm(self.visual_proj(visual_emb)) + self.modality_embed[2]
        fused = self._fuse_tokens(torch.stack([ocr_h, trans_h, vis_h], dim=1))   # (B, 3, H) -> (B, H)
        return nn.functional.normalize(self.out_proj(fused), dim=-1)
```

**Why each piece:**
- **Linear projections:** the three inputs have different sizes (384, 384, CLIP); attention needs one shared width.
- **Z-score normalization:** text embeddings have larger magnitudes than visual ones. Without it, the biggest modality dominates the attention scores.
- **Modality-type embeddings:** attention is permutation-invariant. Without a type embedding, the model can't tell the transcript token from the slide token.
- **Attention over only 3 tokens:** nearly free (a 3×3 attention matrix) and lets the model weight modalities per segment.
- **L2-normalized output:** so the FAISS inner product equals cosine similarity.

**The pairwise variant** (`forward_with_parts`) runs each modality through its own residual expert, fuses each of the 3 pairs with the *same* block, averages them, and also returns each pair's embedding so training can add a contrastive loss per pair:

```python
h = [x + expert(x) for x, expert in zip(h, self.experts)]               # unimodal experts (residual)
pair_h = {name: self._fuse_tokens(torch.stack([h[a], h[b]], dim=1))     # shared block for every pair
          for name, (a, b) in self.PAIRS.items()}                       # ocr+asr, asr+vis, ocr+vis
z = normalize(self.out_proj(torch.stack(list(pair_h.values())).mean(0)))
```

A **query projector** (2-layer MLP) maps the Sentence-BERT query embedding into the same 384-d space, so `score(query, segment) = cos(QueryProjector(e_query), z_seg)`.

### Training: symmetric InfoNCE with in-batch negatives (`train.py`)

```python
# src/stage2_retrieval/train.py
def info_nce_loss(query_z, seg_z, temperature, seg_key=None):
    logits = query_z @ seg_z.t() / temperature                 # (B, B): every query vs every segment
    labels = torch.arange(logits.size(0), device=logits.device)  # the diagonal is the right answer
    if seg_key is not None:
        same = seg_key.unsqueeze(0) == seg_key.unsqueeze(1)    # two queries for the SAME segment
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))       # are not negatives: mask them out
    loss_q2s = F.cross_entropy(logits, labels)                 # query -> segment
    loss_s2q = F.cross_entropy(logits.t(), labels)             # segment -> query
    return 0.5 * (loss_q2s + loss_s2q)
```

- **In-batch negatives:** in a batch of B pairs, the other B−1 segments are free negatives; no mining is needed.
- **Temperature τ = 0.07:** sharpens the softmax so the model is pushed to separate close candidates.
- **Symmetric:** both directions are trained, so the space works for query→segment and segment→query.
- **Masking false negatives:** two pseudo-queries from the same segment in one batch would otherwise push a segment away from itself.
- **Pseudo-queries:** without human questions, training uses each segment's most informative sentence, its top keywords and its slide title (filler removed, duplicates dropped). These are **never** used for evaluation, because they're built from the segment's own text, so finding them is trivial.

**Honest status.** The checkpoint in the repo was trained on pseudo-queries before several data fixes. It's weak (MRR 0.20, Part VII), and retraining on human questions is the top item on the to-do list. The `auto` retrieval mode (3.6) exists precisely so a weak model can't hurt the app.

## 3.5 Indexing (`src/rag/build_index.py`)

```bash
python -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05
```

For each lecture:
1. **Load features**, or extract them with `--extract-missing`. Old-format features without raw Whisper segments trigger a warning.
2. **Run Stage 1** to get the segments.
3. **Build the segment texts and the three embeddings** (`segment_bundle`).
4. **Run Stage 2** to get the fused vector per segment.
5. **Drop segments outside the annotated `lecture_span`.** This removes pre-class chatter and the roll call (privacy).
6. **Attach display metadata** from `data/video_meta.json` (title, URL).

```python
# src/rag/build_index.py (trimmed)
stage1 = load_stage1_model(args.stage1_checkpoint, args.device)
stage2 = load_stage2_model(args.stage2_checkpoint, args.device)
text_enc = TextEncoder(device=args.device)
for vid in ids:
    ensure_features(vid, args.videos, args.extract_missing, args.device)
    feats = load_features(os.path.join(PATHS.features_dir, f"{vid}.npz"))
    segs = segment_video(vid, stage1, PATHS.features_dir, args.device)
    b = segment_bundle(feats, segs, text_enc)
    with torch.no_grad():
        z = stage2.encode_segments(*(torch.from_numpy(b[k].astype(np.float32))
                                     for k in ("ocr_emb", "transcript_emb", "visual_emb")))
    ...  # one record per segment: id, video, title, url, start/end, transcript, slide text
```

It then writes:

| File | Content | Why |
|---|---|---|
| `segments.index` | FAISS `IndexFlatIP` over the fusion embeddings | Exact cosine search (vectors are normalized) |
| `segments_metadata.json` | segment ID, video, title, URL, start/end, full transcript, slide text | Everything needed for answers, citations and BM25 |
| `text_embeddings.npy` | Sentence-BERT embeddings of each segment's text | The app starts in seconds instead of ~30 s of encoding |
| `index_info.json` | build time, checkpoints, videos | Reproducibility |
| `thresholds.json` | refusal thresholds, written later by `eval/run.py` | The gate uses calibrated, not hand-picked, values |

**Current index: 157 segments from 5 lectures, built in about 63 s, about 0.9 MB on disk.** The fact that the whole searchable knowledge of five hours of video fits in under 1 MB is what makes the deployment in Part IV possible.

## 3.6 Retrieval

| Retriever | How it scores | Strong at | Weak at |
|---|---|---|---|
| **BM25** | term frequency (saturating, k1) × inverse document frequency, length-normalized (b) | exact technical terms, no training | synonyms and paraphrases |
| **Dense (Sentence-BERT `all-MiniLM-L6-v2`)** | cosine between query and segment embeddings | paraphrases, meaning | rare tokens, ASR misspellings |
| **Fusion (Stage 2)** | FAISS inner product with the learned encoder | in principle, multimodal evidence | currently weakly trained |
| **Hybrid** | top 2k from BM25 and dense; BM25 divided by its max (0–1); `0.5 · BM25 + 0.5 · cosine` | covers both failure modes | — |
| **`auto`** (the original app mode) | fusion scores, but falls back to hybrid if they look degenerate | robust to an untrained model | — |

### Hybrid (`src/stage2_retrieval/retrieval.py`)

```python
# LectureRetriever._combined_fallback_search
text_results = self._text_similarity_search(query, top_k * 2)
bm25_results = self._bm25_search(query, top_k * 2)

combined = {}
for r in text_results:
    key = (r["video_id"], r["start_time"], r["end_time"])
    combined[key] = {**r, "text_score": r["score"], "bm25_score": 0.0}

# BM25 is unbounded: scale it to [0, 1] by the best score for this query
bm25_scores = [r["score"] for r in bm25_results] if bm25_results else [0.0]
bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
for r in bm25_results:
    key = (r["video_id"], r["start_time"], r["end_time"])
    norm_score = r["score"] / bm25_max
    if key in combined:
        combined[key]["bm25_score"] = norm_score
    else:
        combined[key] = {**r, "text_score": 0.0, "bm25_score": norm_score}

for key, r in combined.items():                         # 50% semantic + 50% keyword
    r["score"] = 0.5 * r["text_score"] + 0.5 * r["bm25_score"]
return sorted(combined.values(), key=lambda x: -x["score"])[:top_k]
```

**Why max-normalize BM25:** BM25 scores have no upper bound (they grow with query length and term rarity), while cosine is in [−1, 1]. Adding them raw would let BM25 dominate. Reciprocal-rank fusion (RRF) is the scale-free alternative and is used in the research harness.

### The `auto` mode's variance trigger

```python
# LectureRetriever.search, mode="auto" (trimmed)
scores_arr = [r["score"] for r in faiss_results]
avg_faiss_score, variance = np.mean(scores_arr), np.var(scores_arr)
# Untrained (all low), collapsed (all high / all the same): don't trust the model
if avg_faiss_score < 0.15 or avg_faiss_score > 0.85 or variance < 0.01:
    return self._combined_fallback_search(query, top_k)
# Otherwise: model scores boosted by text similarity
r["score"] = 0.7 * r["score"] + 0.3 * text_boost
```

This was added on Day 7, when the first InfoNCE model **collapsed** (every segment got nearly the same score). It keeps the app useful whatever state the learned model is in. Measured today, `auto` gives exactly the hybrid numbers, because the weak fusion model always trips the trigger.

### Cross-encoder reranker (`src/retrieval/reranker.py`)

Bi-encoders and BM25 never look at the question and the segment *together*. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads `[question][SEP][segment]` jointly through one transformer and outputs a relevance logit. It is much more precise but too slow for the whole corpus, so it only re-scores the **top 20** candidates.

```python
# src/retrieval/reranker.py
class CrossEncoderReranker:
    def __init__(self, model_name=MODELS.cross_encoder, device="cpu", max_chars=1500):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.max_chars = max_chars

    def rerank(self, query, hits, order="transcript_first"):
        def text(h):
            a, b = h.get("transcript_text", ""), h.get("ocr_text", "")
            return ((a + " " + b) if order == "transcript_first" else (b + " " + a))[: self.max_chars]
        scores = self.model.predict([(query, text(h)) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
            h["rerank_prob"] = 1.0 / (1.0 + math.exp(-float(s)))     # sigmoid -> 0..1 confidence
        return sorted(hits, key=lambda h: -h["rerank_score"])
```

**Input order matters.** The model reads at most 512 tokens. The original version put the noisy OCR text first, so most of the window was OCR garbage plus the first sentences of the transcript. Transcript-first lifted MRR from 0.556 to 0.666 (Story 2).

### The search service and confidence (`src/retrieval/search.py`)

`SearchService` wraps the existing `LectureRetriever` unchanged and adds reranking and a **confidence score** for the refusal gate:

```python
# src/retrieval/search.py (trimmed)
def dense_confidence(self, query: str) -> float:
    """Best cosine between the question and ANY segment in the collection."""
    q = self.retriever.text_encoder.encode([query])[0]
    q = q / (np.linalg.norm(q) + 1e-8)
    return float(np.max(self.retriever.segment_text_embeddings @ q))

def search(self, query, top_k=5, mode="hybrid", rerank=None, candidates=20, rerank_order="transcript_first"):
    rerank = (self.reranker is not None) if rerank is None else rerank
    hits = self.retriever.search(query, top_k=candidates if rerank else top_k, mode=mode)
    for h in hits:                                   # attach full metadata (times, texts, title, url)
        h.update({k: v for k, v in self.by_id.get(h["segment_id"], {}).items() if k not in h})
        h["retrieval_score"] = h["score"]
    conf, source = self.dense_confidence(query), "dense"
    if rerank:
        hits = self.reranker.rerank(query, hits, order=rerank_order)
        if hits:
            conf, source = hits[0]["rerank_prob"], "rerank"   # top reranker probability
    return SearchResult(hits=hits[:top_k], confidence=conf, confidence_source=source, timings_ms=timings)
```

- **Without the reranker:** confidence is the best cosine similarity between the question and *any* segment.
- **With the reranker:** it's the sigmoid of the top reranker logit.
- **Thresholds** are loaded from `<index>/thresholds.json` (written by the evaluation), and can be overridden with `RAG_MIN_CONFIDENCE`.

## 3.7 The RAG pipeline (`src/rag/`)

The whole online flow is one method, `RAGPipeline.ask()`. Reading it top to bottom is the fastest way to understand the system:

```python
# src/rag/pipeline.py (trimmed)
REFUSAL = "This isn't covered in the indexed lectures."

def ask(self, question, history=None, use_reranker=None, top_k=None) -> AskResponse:
    query = rewrite_query(question, history or [], self.llm)                       # 1. follow-ups
    res = self.search.search(query, top_k=top_k or 5, mode=self.mode,
                             rerank=use_reranker, candidates=20)                    # 2. retrieve + rerank
    threshold = self.search.threshold(res.confidence_source)

    if not res.hits or res.confidence < threshold:                                  # 3. refusal gate
        return respond(REFUSAL, [], "refused", refused=True)                        #    (no LLM call)

    context, labels = build_context(res.hits)                                       # 4. [S1]..[S5]
    if self.llm is None:
        return respond("No LLM is configured ...", [make_citation(l, h) for l, h in labels.items()],
                       "retrieval_only")

    try:
        raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, context))  # 5. LLM (JSON)
    except LLMError as e:
        return respond(f"The language model is unavailable ({e}) ...", all_citations, "llm_error")

    parsed, err = parse_llm_output(raw)                                             # 6. Pydantic parse
    if parsed is None:
        return respond("The model's answer could not be parsed ...", all_citations, "llm_error")
    if not parsed.covered:
        return respond(REFUSAL, [], "refused", refused=True)

    text, cites, dropped = validate_citations(parsed, labels)                       # 7. check citations
    if not cites:                                    # an answer with no valid citation is not grounded
        return respond(REFUSAL, [], "refused", refused=True, dropped=dropped)
    return respond(text, cites, "rag", dropped=dropped)                             # 8. respond
```

**The four modes** a response can have: `rag` (a grounded answer), `retrieval_only` (no key), `refused` (gate, `covered=false`, or no valid citation) and `llm_error` (provider failure or unparseable output). Every response carries per-stage latency.

### Step 1: follow-up rewrite (`rewrite.py`)

```python
# src/rag/rewrite.py
# Words/phrases that point back to the previous turn. Deliberately NOT plain
# "why"/"how"/"example": "Why is softmax used?" is a standalone question.
FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|the same|the above|the previous)\b"
    r"|\b(more simply|in simpler|simpler terms|explain more|tell me more|more detail|elaborate|"
    r"again|another example|give an example|an example of it|how so|why is that|what about)\b", re.I)

def is_follow_up(question: str) -> bool:
    """Short questions ("and RNNs?") or ones that refer back to the last turn."""
    return len(question.split()) <= 4 or bool(FOLLOW_UP.search(question))

def rewrite_query(question, history, llm=None, max_turns=3) -> str:
    users = [t.content for t in history if t.role == "user"][-max_turns:]
    if not users or not is_follow_up(question):
        return question
    if llm is not None and os.environ.get("RAG_LLM_REWRITE") == "1":
        ...  # optional: ask the LLM to rewrite; falls back to the heuristic on any error
    return " ".join(users[-2:] + [question])
```

- **Why prepend instead of rewrite:** "explain it more simply" has no searchable content. Prepending the last two user questions gives retrieval the topic; the LLM still sees the original wording.
- **Why deterministic by default:** it's free, instant and unit-testable. An LLM rewrite costs a second call against the rate limit.
- **A real bug fixed here (Day 28):** the first regex contained "why", "how" and "example". So *"Why is softmax used at the output layer?"*, asked after a question about pagination, was treated as a follow-up and became "pagination … Why is softmax …". Lesson: a follow-up must *refer back*, not merely be a question. Tests now cover both directions.

### Step 4: context builder (`context.py`)

```python
# src/rag/context.py
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1          # rough, but good enough for budgeting

def dedupe(hits, max_overlap=0.5):
    kept = []
    for h in hits:                                     # hits arrive best-first
        if all(_overlap_frac(h, k) <= max_overlap for k in kept):
            kept.append(h)
    return kept

def build_context(hits, max_tokens=1800, per_segment_tokens=450):
    blocks, labels, used = [], {}, 0
    for i, h in enumerate(dedupe(hits), start=1):
        label = f"S{i}"
        header = f"[{label}] {h['title']} — {format_timestamp(h['start_time'])}–{format_timestamp(h['end_time'])}"
        budget = min(per_segment_tokens, max_tokens - used - estimate_tokens(header))
        if budget < 40:
            break                                      # out of budget: drop the lowest-ranked segments
        words = ("Slide text: " + h["ocr_text"] + "\n" if h["ocr_text"] else "") + "Transcript: " + h["transcript_text"]
        block = header + "\n" + " ".join(words.split(" ")[: int(budget / 1.3)])
        blocks.append(block); labels[label] = h; used += estimate_tokens(block)
    return "\n\n".join(blocks), labels
```

- **Dedupe:** two overlapping segments from the same lecture waste budget and confuse citations; the higher-ranked one wins.
- **Short labels:** LLMs copy `[S2]` reliably; long IDs like `college_lec03#026` get mangled. The `labels` dict maps them back.
- **Budget (1,800 total, 450 per segment):** keeps every request well under Groq's per-minute token limit and leaves room for the answer.

### Step 5: the prompt (`prompts.py`)

```text
You are a teaching assistant for a university course. You answer students' questions using ONLY the
lecture segments provided to you. Each segment is labelled [S1], [S2], ... and contains the slide text
and the transcript of what the instructor said (the transcript comes from speech recognition and may
contain errors).

Rules:
1. Use only information in the provided segments. Do not add facts from your own knowledge.
2. End EVERY sentence with the label(s) of the segment(s) that support it, for example:
   "Softmax turns the network's raw scores into probabilities [S1]." Never write a sentence without a
   label; if no segment supports a sentence, leave that sentence out.
3. If the segments do not contain the answer, set "covered" to false and say briefly that the indexed
   lectures do not cover it.
4. Be clear and concise (at most about 150 words). Explain in plain language suitable for a student.

Return ONLY a JSON object with exactly these keys:
{"answer": "<answer text with [S#] citations>", "citations": ["S1", ...], "covered": true}
```

- **Rule 2 was strengthened on Day 28.** The first version said "cite every factual sentence". With `gpt-oss-20b` that produced answers with citations only at the end and one sentence of outside knowledge. "End EVERY sentence… if no segment supports a sentence, leave it out", plus a concrete example, fixed it.
- **Warning the model that transcripts contain ASR errors** stops it from "correcting" a misheard word into a different fact.
- **`temperature = 0.0`** in every provider: we want the most likely, repeatable answer, not creativity.

### Step 6: tolerant parsing (`schema.py`)

```python
# src/rag/schema.py (trimmed)
class LLMAnswer(BaseModel):
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    covered: bool = True

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v):                     # "S1" -> ["S1"], ["[S1]", 2] -> ["S1", "2"]
        if v is None:
            return []
        if isinstance(v, (str, int)):
            v = [v]
        return [str(x).strip().strip("[]") for x in v if str(x).strip()]


def parse_llm_output(text):
    if not text or not text.strip():
        return None, "empty output"
    cleaned = re.sub(r"```(?:json)?", "", text).strip()            # strip code fences
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", cleaned)]:  # try every "{" until one parses
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            try:
                return LLMAnswer.model_validate(obj), None
            except ValidationError as e:
                return None, f"schema error: {e.errors()[0]['msg']}"
    return None, "no JSON object found"
```

**Why `raw_decode` from every `{`:** models sometimes write "Here is the JSON: {...} Hope this helps!". `raw_decode` parses one object and ignores what follows, so prose before or after doesn't break parsing. It **returns** an error instead of raising, so the pipeline can degrade to `llm_error` mode.

### Step 7: citation validation (`citations.py`)

```python
# src/rag/citations.py
LABEL_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")      # [S1] or [S1, S3]

def validate_citations(parsed, labels):
    """Returns (answer text with readable citations, valid citations, dropped labels)."""
    cited = list(dict.fromkeys(parsed.citations + [x.strip() for m in LABEL_RE.findall(parsed.answer)
                                                    for x in m.split(",")]))
    valid = [c for c in cited if c in labels]
    dropped = [c for c in cited if c not in labels]           # hallucinated sources

    def repl(m):                                              # [S1] -> [Lecture 1, 48:33]
        keep = [x.strip() for x in m.group(1).split(",") if x.strip() in labels]
        return " ".join(f"[{labels[x]['title']}, {format_timestamp(labels[x]['start_time'])}]" for x in keep)

    # A sentence whose ONLY citations were invalid is unsupported: remove it.
    kept_sentences = []
    for sent in re.split(r"(?<=[.!?])\s+", parsed.answer.strip()):
        cited_here = [x.strip() for m in LABEL_RE.findall(sent) for x in m.split(",")]
        if cited_here and not any(x in labels for x in cited_here):
            continue
        kept_sentences.append(sent)
    text = re.sub(r"\s+([.,;:])", r"\1", LABEL_RE.sub(repl, " ".join(kept_sentences))).strip()
    return text, [make_citation(c, labels[c]) for c in valid], dropped


def video_link(hit):
    start = int(hit["start_time"])
    url = hit.get("url") or f"/videos/{hit['video_id']}.mp4"
    if "youtu" in url:
        return url + ("&" if "?" in url else "?") + f"t={start}s"     # YouTube: &t=765s
    return f"{url}#t={start}"                                          # HTML5 media fragment: #t=765
```

**This is where timestamps come from.** The LLM never writes a time; it only writes `[S1]`. The code looks up S1's segment and reads its `start_time` from the index metadata, so a timestamp can't be hallucinated. It is exactly as accurate as the segment boundary.

**Limit to know:** a sentence with *no* label at all is kept (the prompt forbids it, but code can't prove a sentence is supported). Faithfulness (3.10) measures how often that matters.

## 3.8 LLM providers (`src/llm/`)

**One interface.** The pipeline never knows which model is behind it:

```python
# src/llm/base.py
class LLMClient(ABC):
    name: str = "base"
    model: str = ""
    last_stats: Dict = {}  # filled by HTTP providers: request_ms, attempts, rate_limited

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        """Return the model's raw text answer."""
```

**Implementations** (`providers.py`): **Groq** (OpenAI-compatible chat API, default `openai/gpt-oss-20b`), **Gemini** (REST, `gemini-2.5-flash`), **Ollama** (local, `llama3.2:3b`) and a **Mock** with scripted outputs for tests. `factory.get_llm_client()` picks one from `LLM_PROVIDER` (or auto-detects from which key is set) and returns `None` when nothing is configured, which switches the pipeline to retrieval-only mode.

### Robust HTTP: `post_json`

```python
# src/llm/base.py (trimmed)
def post_json(url, payload, headers=None, timeout=30.0, retries=None, backoff=1.5, provider="LLM", stats=None):
    retries = int(os.environ.get("LLM_RETRIES", "3")) if retries is None else retries
    stats = stats if stats is not None else {}
    stats.update({"attempts": 0, "rate_limited": 0, "request_ms": None})
    for attempt in range(retries + 1):
        stats["attempts"] = attempt + 1
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        except requests.Timeout:
            last_err = LLMTimeoutError(f"{provider} did not respond within {timeout:.0f}s")
        except requests.RequestException as e:
            last_err = LLMError(f"Could not reach {provider}: {e}")
        else:
            if r.status_code == 200:
                stats["request_ms"] = (time.perf_counter() - t0) * 1000    # model time of the successful try
                return r.json()
            if r.status_code in (401, 403):
                raise LLMError(f"{provider} rejected the API key ...")      # retrying won't help
            if r.status_code == 429:
                stats["rate_limited"] += 1
                last_err = RateLimitError(f"{provider} rate limit hit (HTTP 429) ...")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < retries:
                    time.sleep(min(float(retry_after), 30.0))               # the server told us how long
                    continue
            elif r.status_code >= 500:
                last_err = LLMError(f"{provider} server error (HTTP {r.status_code}).")
            else:
                raise LLMError(f"{provider} request failed (HTTP {r.status_code}): {r.text[:300]}")
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, 0.5))   # exponential backoff + jitter
    raise last_err
```

- **Which errors are retried:** 429 (rate limit), 5xx (server), timeouts and connection errors are *transient*. 401/403 (bad key) and other 4xx are *permanent*, so they fail immediately.
- **Exponential backoff** (1.5 s, 3 s, 6 s…) gives the provider time to recover; **jitter** stops many clients retrying in lockstep.
- **`Retry-After`** is honoured when the server sends it, because the server knows best.
- **`stats`** separates the model's own response time (`request_ms`) from time lost to retries. This was added on Day 28, when the first evaluation reported a 23.7 s "LLM latency" that was mostly rate-limit waiting.

### Surviving retired models (Groq)

Groq retires models regularly; `llama-3.1-8b-instant` disappeared in 2026 and every request started failing with `404 model_not_found`.

```python
# src/llm/providers.py (trimmed)
class GroqClient(LLMClient):
    URL = "https://api.groq.com/openai/v1/chat/completions"
    MODELS_URL = "https://api.groq.com/openai/v1/models"
    PREFERRED = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b",
                 "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def complete(self, system, user, json_mode=True):
        try:
            return self._complete(system, user, json_mode)
        except LLMError as e:
            if "model_not_found" not in str(e) and "decommissioned" not in str(e):
                raise
            replacement = self._pick_available_model()          # GET /models, first preferred one that exists
            if not replacement:
                raise LLMError(f"Groq model '{self.model}' is not available ... Set LLM_MODEL in .env") from e
            print(f"[GroqClient] Model '{self.model}' is unavailable; switching to '{replacement}'.")
            self.model = replacement
            return self._complete(system, user, json_mode)

    def _complete(self, system, user, json_mode=True):
        payload = {"model": self.model, "temperature": 0.0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        self.last_stats = {}
        data = post_json(self.URL, payload, {"Authorization": f"Bearer {self.api_key}"}, _timeout(),
                         provider="Groq", stats=self.last_stats)
        return data["choices"][0]["message"]["content"]
```

**Why this design:** a free demo that can die overnight because a vendor renamed a model is not a demo. The client asks Groq which models exist and picks the first from a preference list. It only does that on the specific "model not found" error, so other failures still surface.

## 3.9 Serving: API and demo

### FastAPI (`app/backend/main.py`)

```python
# app/backend/main.py (trimmed)
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    rag = get_rag()                                        # built once, lazily, then reused
    if rag is None:
        raise HTTPException(status_code=503, detail=f"RAG pipeline not available: {_rag_error}. "
                                                    f"Build it with: python -m src.rag.build_index")
    return rag.ask(req.question, history=req.history, use_reranker=req.use_reranker, top_k=req.top_k)


@app.get("/health")
def rag_health():
    rag = get_rag()
    return {"status": "ok" if rag is not None else "degraded",
            "rag_index_loaded": rag is not None,
            "num_segments": len(rag.search.retriever.metadata) if rag else 0,
            "reranker": rag is not None and rag.search.reranker is not None,
            "llm": f"{rag.llm.name}:{rag.llm.model}" if rag and rag.llm else None,
            "mode": ("rag" if rag and rag.llm else "retrieval_only") if rag else None,
            "error": _rag_error}
```

- **Request validation is free:** `AskRequest` limits the question to 1–1,000 characters and `top_k` to 1–10. FastAPI returns a 422 automatically for anything else.
- **`response_model=AskResponse`** guarantees the response shape and generates the OpenAPI docs at `/docs`.
- **503 rather than 500** when the index is missing: the service is up but not ready, and the message says how to fix it.
- The original `GET /api/search`, `/api/health` and `/api/stats` and the HTML frontend are unchanged; raw videos are served under `/videos`.

### Gradio demo (`space/app.py`)

- **Layout:** a question box with example questions; the answer; citation cards; a "Show retrieved segments" table with scores; a reranker switch; a latency line; follow-up memory and a New conversation button.
- **One file runs locally and on the Space.** It works out where it is running and finds the index:

```python
# space/app.py (trimmed)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "src")) else os.path.dirname(HERE)   # Space root or repo root
sys.path.insert(0, ROOT)
if not os.environ.get("RAG_INDEX_DIR"):
    local = os.path.join(ROOT, "rag_index")                                           # packaged Space
    os.environ["RAG_INDEX_DIR"] = local if os.path.isdir(local) else os.path.join(ROOT, "data", "rag_index")

SEARCH = SearchService(index_dir=os.environ["RAG_INDEX_DIR"], use_reranker=True)
SEARCH.warm_up()                                       # load models now, not on the first user's click
```

- **The player for each citation** picks the best available source:

```python
def player(c) -> str:
    start = int(c.start_time)
    if "youtu" in c.url:                                           # 1. YouTube link: embed at the second
        return youtube_embed(c.url, start) or f'<a href="{c.url}" target="_blank">Open video at {c.timestamp}</a>'
    local = os.path.join(VIDEOS_DIR, f"{c.video_id}.mp4")
    if os.path.exists(local):                                      # 2. local file: HTML5 player at #t=
        src = f"/gradio_api/file={local.replace(os.sep, '/')}#t={start}"
        return f'<video width="100%" height="220" controls preload="metadata" src="{src}"></video>'
    return (f"<i>📍 {html.escape(c.display)}: the recordings are not public, so open this lecture "
            f"at the timestamp in your course material.</i>")       # 3. deployed demo: text only
```

- **Windows noise fix.** On Windows, a browser dropping a keep-alive connection makes asyncio's Proactor loop print a scary `ConnectionResetError [WinError 10054]` traceback. It's harmless, so the app wraps `_call_connection_lost` to swallow exactly that error, and only on Windows.
- **HTML escaping:** snippets come from transcripts, so they are escaped with `html.escape` before going into HTML.

## 3.10 Evaluation (`eval/`)

### The question set (`eval/make_questions.py`)

| Group | Count | Source | Notes |
|---|---|---|---|
| Drafted | 49 | written from the transcripts of lectures 1–5 | must be human-verified; tend to reuse lecture wording, which favours BM25 |
| Human pilot | 17 | `data/annotations/gold/` (lectures 1–3) | written by a person without looking at the transcript wording |
| Out-of-scope | 10 | topics the course never covers | key terms are automatically checked to be **absent** from the index |

### Gold labels are time spans, not segment IDs (`eval/gold.py`)

```python
# eval/gold.py
def gold_segments(q, metadata, min_frac=0.5):
    out = set()
    for m in metadata:
        if m["video_id"] != q["video_id"]:
            continue
        inter = max(0.0, min(m["end_time"], q["end"]) - max(m["start_time"], q["start"]))
        if inter <= 0:
            continue
        seg_len = max(1e-6, m["end_time"] - m["start_time"])
        span_len = max(1e-6, q["end"] - q["start"])
        # gold if the overlap covers half the SEGMENT, or half the ANSWER SPAN
        if inter / seg_len >= min_frac or inter / span_len >= min_frac:
            out.add(m["segment_id"])
    return out
```

**Why:** segment IDs change every time lectures are re-segmented; time spans don't. The two-sided rule handles a long answer spread over several short segments as well as a short answer inside one long segment.

### Retrieval metrics (`eval/metrics.py`)

```python
def recall_at_k(ranked, gold, k):  return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0

def mrr(ranked, gold):
    for i, s in enumerate(ranked, start=1):
        if s in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], start=1) if s in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0

def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)   # resample questions
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
```

| Metric | Question it answers |
|---|---|
| Recall@5 / @10 | Of the right moments, how many are in the top 5 / 10? |
| MRR | How high is the *first* right moment? (1 = first, 0.5 = second…) |
| nDCG@10 | Are the right moments ranked high overall, with a log discount by rank? |
| Bootstrap 95% CI | How much would the number move with a different sample of questions? With 66 questions, a lot. |

### Refusal calibration (`eval/run.py`)

```python
# eval/run.py (trimmed)
def calibrate(search, qs, rerank):
    """Pick the confidence threshold that maximizes balanced accuracy on dev; report it on test."""
    scored = []
    for q in qs:
        res = search.search(q["question"], top_k=5, mode="hybrid", rerank=rerank)
        scored.append((q, res.confidence, res.confidence_source))

    def bal_acc(items, th):
        ins = [c >= th for q, c, _ in items if q["in_scope"]]      # in-scope should be ANSWERED
        oos = [c < th for q, c, _ in items if not q["in_scope"]]   # out-of-scope should be REFUSED
        if not ins or not oos:
            return None, None, None
        return (np.mean(ins) + np.mean(oos)) / 2, float(np.mean(oos)), float(1 - np.mean(ins))

    dev = [s for s in scored if dev_test_split(s[0]) == "dev"]
    test = [s for s in scored if dev_test_split(s[0]) == "test" and s[0]["source"] != "pilot_human"]
    cands = sorted(set(round(c, 3) for _, c, _ in dev))                 # every observed confidence
    best = max(cands, key=lambda th: (bal_acc(dev, th)[0] or 0, -th))   # ties -> lower threshold
    ba, oos_refused, ins_refused = bal_acc(test, best)                  # reported on TEST only
```

- **Balanced accuracy, not accuracy:** there are far more in-scope than out-of-scope questions, so "never refuse" would score high on plain accuracy.
- **Dev/test split:** a deterministic hash of the question ID. Choosing the threshold on the questions you report it on would overstate it.
- **Ties go to the lower threshold:** refusing a real question is worse for a student than attempting a borderline one.

### Generation metrics (needs an API key)

- **Citation accuracy:** the share of grounded (`mode == "rag"`) answers that cite at least one gold segment. Error fallbacks don't count.
- **Faithfulness:** an LLM-as-judge (`eval/judge.py`) sees **only the cited segments** and the answer, splits the answer into claims, and labels each supported or unsupported. Faithfulness = supported / total, averaged over answers.
- **End-to-end refusal:** out-of-scope questions refused and in-scope questions wrongly refused, through the whole pipeline.
- **Latency p50/p95** per stage, with `llm_request_ms` (model time) separate from `llm_ms` (including rate-limit waits).
- **Error counts:** LLM errors after retries, rate-limited calls and judge failures are reported, never hidden.

### Research harness (`scripts/`)

`run_evaluation.py` and `run_ablations.py` cover:
- segmenters × retrievers;
- Pk, WindowDiff and boundary F1 (±15 s tolerance) for segmentation;
- paired bootstrap significance tests;
- signal-removal, encoder, fusion-mode, segment-length, length-matched and seed ablations.

They refuse to run on a split with no human gold data.

---

# Part IV: Deployment

## 4.1 Goal and constraints

**Goal:** a link anyone can open with one click, no install, that answers questions about the indexed lectures.

**Constraints:**
1. **Free.** A student project can't pay for servers.
2. **CPU only.** Free hosting has no GPU.
3. **Private recordings.** The videos are classroom recordings showing students. The college gave permission to publish the demo, but **the videos themselves can't be uploaded anywhere public** (not YouTube, not the Space).
4. **The API key must stay secret**, even though the code is public.

## 4.2 Why Hugging Face Spaces

| Option | Verdict |
|---|---|
| **Hugging Face Spaces (Gradio, CPU basic)** | **Chosen.** Free, 2 vCPU and 16 GB RAM, native Gradio support, secrets management, git-based deploys, a stable public URL. |
| Streamlit Community Cloud | Free, but ~1 GB RAM: too small for Sentence-BERT + cross-encoder + FAISS. |
| Render / Railway free tiers | Small RAM and sleep after minutes of inactivity; more setup. |
| A college server | Not available to students; the deadline didn't allow for paperwork. |
| A VPS (AWS / GCP) | Costs money, and we'd be responsible for security patches. |

## 4.3 What makes it deployable at all

The design decision from Day 22 pays off here: **all the heavy work happens offline.** The deployed app needs:
- the **index** (157 segments: transcripts, slide text, embeddings, FAISS), about 0.9 MB;
- the **Stage 2 checkpoint**, about 8 MB;
- the **inference code** (no training, no Whisper, no OCR);
- **three small models** downloaded at start-up from the Hugging Face Hub: Sentence-BERT, the cross-encoder and the tokenizer.

Total upload: **about 9.4 MB.** No video, no GPU, no feature extraction.

## 4.4 The packager: `space/prepare_space.py`

```bash
python space/prepare_space.py --allow-internal-videos --license other
```

It assembles `build/hf_space/` (git-ignored), a folder that is a complete Space:

```
build/hf_space/
├── README.md              # Space card with the YAML header Hugging Face reads
├── app.py                 # copy of space/app.py
├── requirements.txt       # copy of space/requirements.txt (pinned, CPU-only torch)
├── config.py
├── src/...                # only the inference-side modules (an explicit list, SRC_FILES)
├── rag_index/             # the precomputed index
└── checkpoints/stage2_retrieval_model.pt
```

The important parts of the script:

```python
# space/prepare_space.py (trimmed)
meta = json.load(open(os.path.join(args.index, "segments_metadata.json"), encoding="utf-8"))
internal = sorted({m["video_id"] for m in meta if "internal" in m.get("license", "").lower()})
if internal and not args.allow_internal_videos:
    sys.exit(f"Refusing: {internal} are marked internal in data/video_meta.json. ... pass "
             f"--allow-internal-videos only if you have written permission to publish them.")

shutil.rmtree(OUT, ignore_errors=True)
for rel in SRC_FILES:                                   # an allow-list, not "copy everything"
    shutil.copy(os.path.join(ROOT, rel), os.path.join(OUT, rel))
shutil.copytree(args.index, os.path.join(OUT, "rag_index"))
shutil.copy(".../checkpoints/stage2_retrieval_model.pt", ".../checkpoints/")

# Video files are too large for a Space, so citations play from the URLs in
# data/video_meta.json (e.g. unlisted YouTube). Refresh them here so adding
# a link does not require rebuilding the index.
for m in meta:
    url = (vm.get(m["video_id"]) or {}).get("url")
    if url:
        m["url"], m["is_youtube"] = url, "youtu" in url
json.dump(meta, open(os.path.join(OUT, "rag_index", "segments_metadata.json"), "w", encoding="utf-8"))
```

**Design choices:**
- **An allow-list of source files.** Copying only what inference needs means no training scripts, no annotation files, no `.env` and no `data/` folder can leak into a public repo by accident.
- **A privacy guard that defaults to "no".** Every lecture is tagged `internal` in `data/video_meta.json`, so the script refuses to package unless someone explicitly passes `--allow-internal-videos`. The flag is a deliberate "yes, we have permission" step, not a default.
- **The Space README carries the YAML header** Hugging Face reads (`sdk: gradio`, `sdk_version: 5.50.0`, `app_file: app.py`) and an attribution list per lecture.
- **Pinned requirements** (`space/requirements.txt`). Exact versions tested locally, plus `--extra-index-url https://download.pytorch.org/whl/cpu`, so the Space installs the ~200 MB CPU build of PyTorch instead of the ~2 GB CUDA build.

## 4.5 The video decision: text-only citations

The recordings can't be published, so the deployed demo has **no video player**. The options were:

| Option | Decision |
|---|---|
| Ship the `.mp4` files in the Space | No: 4.5 GB is too big for a free Space, and it would publish the recordings. |
| Upload to YouTube as Unlisted and embed at the timestamp | No: the college videos can't be uploaded to YouTube. The code supports it (paste links into `data/video_meta.json` and re-package) for openly licensed lectures. |
| Google Drive links | No: the same publishing problem, and Drive embeds don't reliably seek to a timestamp. |
| Slide snapshots (a cropped image of the projector screen at each segment) | Possible later: about 10 MB, no audio, no faces. Not built yet. |
| **Text-only citations** | **Chosen.** Each citation shows `[Lecture 1, 48:33]`, the exact quote and a note to open that lecture at that time in the course material. |

**What still works without video:** everything that matters for answering. Retrieval, reranking, the refusal gate, the LLM answer and citation validation all run on the index, which contains the transcript and slide text of every segment. The timestamps come from segment metadata, not from the video. The local demo, where the `.mp4` files exist, still shows players at the cited second, so a live presentation can use that.

## 4.6 Secrets and configuration

| Setting | Where | Type | Why |
|---|---|---|---|
| `GROQ_API_KEY` | Space → Settings → Variables and secrets | **Secret** | Encrypted, injected as an environment variable at runtime, never shown in the repo or the logs. |
| `LLM_PROVIDER=groq` | same place | Variable | Not sensitive. |
| `LLM_MODEL` | (optional) | Variable | Only to pin a model; the client already falls back if one is retired. |

The code path is the same as locally: `factory.get_llm_client()` reads environment variables. There is no `.env` in the Space, and none is ever copied (it isn't in `SRC_FILES`).

## 4.7 Step by step

**One-time setup:**

1. **Build the package:**
   ```bash
   .venv\Scripts\python.exe space\prepare_space.py --allow-internal-videos --license other
   ```
   Expected: `Space folder ready: ...\build\hf_space (9.4 MB, 5 lectures)`.
2. **Optional local check of the package** (no key, so retrieval-only):
   ```powershell
   cd build\hf_space
   $env:GRADIO_SERVER_PORT = "7861"; ..\..\.venv\Scripts\python.exe app.py
   ```
   Open http://localhost:7861 and ask a question; you should get lecture moments with text-only citations.
3. **Create the Space:** on huggingface.co choose **New Space** → name `conceptbound` → SDK **Gradio** → hardware **CPU basic (free)** → **Public**.
4. **Add the secrets:** Settings → Variables and secrets → **Secret** `GROQ_API_KEY`, **Variable** `LLM_PROVIDER=groq`.
5. **Create a write token:** huggingface.co/settings/tokens → New token → type **Write**.
6. **Push:**
   ```bash
   cd build/hf_space
   git init -b main
   git remote add origin https://huggingface.co/spaces/<your-username>/conceptbound
   git add . && git commit -m "ConceptBound demo"
   git push -u origin main --force
   ```
   Git asks for a username (your Hugging Face username) and a password (paste the token). `--force` replaces the starter README Hugging Face creates.
7. **Wait for the build** (about 5–10 minutes: pip install, then model downloads). The page shows **Building** → **Running**.
8. **Share** `https://huggingface.co/spaces/<your-username>/conceptbound`.

**Updating the live site:** re-run step 1, then `git add . && git commit -m "Update" && git push` inside `build/hf_space`. The Space rebuilds automatically.

**If the build fails:** the Space's **Logs** tab shows the pip or Python error. The usual causes are a missing file in `SRC_FILES` (an `ImportError`) or a dependency version not available for the Space's Python.

## 4.8 How the deployed app behaves

| Situation | What the visitor sees |
|---|---|
| Normal question | A short answer with `[Lecture N, mm:ss]` citations, quotes and text-only location notes, in about 3–4 s. |
| Off-topic question | "This isn't covered in the indexed lectures." |
| Groq rate limit (8,000 tokens per minute on the free tier, shared by all visitors) | After retries: "The language model is unavailable…", followed by the most relevant moments. |
| Groq retires the model | The client switches to another available model automatically. |
| No visitors for about 48 hours | The free Space sleeps; the next visitor waits about a minute while it wakes up. |
| First visit after a rebuild | Model downloads on start-up; `warm_up()` loads them before the UI accepts questions. |

## 4.9 Deployment checklist

- [x] The index and checkpoint are small enough to ship (9.4 MB).
- [x] No video files and no `.env` in the package (allow-list).
- [x] Explicit opt-in flag for internal lectures (written permission obtained).
- [x] The API key is a Space **secret**, not a file.
- [x] CPU-only PyTorch; pinned versions.
- [x] Retrieval-only fallback if the key is missing or the LLM fails.
- [x] The package was tested locally on its own (port 7861) before pushing.
- [ ] Space created and pushed (needs the owner's Hugging Face account).
- [ ] A short "About" note on the Space page saying which course and lectures are indexed.

---

# Part V: The A–Z of ConceptBound

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
| **Y** | **Yield of the gate (refusal)** | The confidence gate refuses before the LLM is called; the threshold is calibrated on dev questions. | Prevents confident nonsense on off-topic questions, and saves an LLM call. |
| **Z** | **Z-score normalization** | Each modality's projection is normalized before fusion. | Stops one modality (bigger magnitude) from dominating attention. |

---

# Part VI: How it was built, day by day

Each "Day" is one focused working session. Dates come from the git history.

## Phase 1: Prototype (13 June – 1 October 2026)

### Day 0: Environment and scaffold (13 June)
- **Built:**
  - a virtual environment;
  - dependencies (PyTorch, transformers, sentence-transformers, Whisper, EasyOCR, OpenCV, FAISS, FastAPI);
  - a folder skeleton (`src/`, `data/`, `checkpoints/`, `app/`, `scripts/`);
  - a central `config.py` with dataclasses (`FEATURES`, `STAGE1`, `STAGE2`, `EVAL`, `MODELS`);
  - a `.gitignore` for videos, data and checkpoints.
- **Why:** one source of truth for paths and hyperparameters avoids "works on my machine" drift across a dozen scripts. `LECTURE_OUTPUT_ROOT` lets outputs live on another drive.

### Day 1: Problem scoping and literature
- **Problem statement:** *given a long lecture, find the moment that answers a free-text student question.*
- **Literature read:** topic segmentation (TextTiling), video–language retrieval (CLIP-based retrievers), lecture indexing (OCR + ASR search).
- **Decision:** a **two-stage design**. Stage 1 decides *where* concepts begin and end; Stage 2 decides *what* each segment is about, so questions can be matched.
- **Base paper:** Arora et al., *Text-to-Multimodal Retrieval with Bimodal Input Fusion in Shared Cross-Modal Transformer* (LREC-COLING 2024). It motivates fusing several modalities for text-to-video retrieval.

### Day 2: Feature extraction
- **Built:**
  - frame sampling every 3 s with OpenCV;
  - Whisper transcription;
  - topic drift from Sentence-BERT on rolling windows;
  - EasyOCR on frames with a text-change signal;
  - optical flow;
  - CLIP embeddings.
  Everything is saved per video as `.npz`.
- **Why 3 s:** fine enough to place boundaries within a few seconds, coarse enough to keep a one-hour lecture at about 1,300 steps (cheap to model).
- **Why `.npz`:** NumPy's compressed format loads in milliseconds and keeps arrays of different shapes in one file.

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
- **Observed:** the InfoNCE model **collapsed** early with too little training data (all segments scored about the same).
- **Response:** the **variance trigger** that falls back to hybrid text search.

### Day 8: API and frontend
- **Built:** FastAPI `GET /api/search`; a single-page HTML frontend with "jump to moment" links; raw videos served under `/videos`.

### Day 9: Evaluation framework
- **Built:** boundary F1 and IoU; Recall@k; mean IoU@1; an evaluation script against baselines.

### Day 10: Ablation hooks and polish (13 July, 1 September and 1 October)
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
- **Fix:**
  - download the weights;
  - make backend loading **raise**;
  - add **screen detection** (median frame → Otsu → largest screen-shaped blob, per 5-minute chunk) so the burnt-in clock isn't read;
  - record provenance (engine, crop source, coverage).
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
- **Annotation tooling.** Transcript sheets, gold templates, a validator with inter-annotator agreement, and annotation guidelines (`docs/ANNOTATION_GUIDELINES.md`).

### Day 15: Verification
- An **end-to-end smoke test** on a scratch copy with synthetic gold files: splits → refuse to evaluate without gold → train → segment → index → evaluate with every baseline.
- The first full re-extraction run started in the background.

## Phase 3: Paper and ablations (16 October – 4 November 2026)

### Day 16: Choosing a venue
- Researched conference deadlines and scope; target: **IEEE ICALT 2027** (track AISLE, 5-page limit, deadline 15 January 2027).
- **Why ICALT:** it's the IEEE conference on advanced learning technologies, so the audience cares about the *learning* problem (students finding explanations), not only retrieval scores. A 5-page full paper suits a system paper with a pilot evaluation.

### Day 17: Paper draft
- **Built:**
  - a Word document generator reproducing the IEEE conference layout;
  - figures: a pipeline diagram, and real signals of one lecture with annotated answer spans;
  - 22 references, all checked against publisher or Crossref records, with Arora et al. as reference [1];
  - a guide version and an anonymized version.
- **Rule kept:** results stay as yellow placeholders until measured.

### Day 18: Pilot annotation
- **3 lectures and 17 human-written queries** (median answer span 123 s), loaded as gold files.
- **Pilot split:** the annotated lectures are the test set.

### Day 19: Base-paper fusion and ablations (28 October)
- **Built:** the `pairwise` fusion mode (unimodal experts, a shared pairwise fusion block, per-pair losses), `--drop-signals`, and `run_ablations.py` (signals, encoder, fusion, segment length, length-matched control, seeds).

### Day 20: Measurement tools (4 November)
- **Built:**
  - `measure_runtime.py`: hardware, extraction time, query latency;
  - `asr_wer.py`: Whisper word error rate against typed reference clips;
  - `collect_paper_results.py`: every paper number in one JSON;
  - a data-driven paper generator;
  - `run_paper_pipeline.py`: everything after extraction, in one command.

## Phase 4: The RAG layer (4 – 6 November 2026)

### Day 21: Audit and plan
- **Findings:** the stale index was built from broken features; the retriever re-encoded every segment at startup (~30 s); there was no LLM key on the development machine.
- **Plan:** reuse the existing retrieval, and add `src/retrieval`, `src/rag`, `src/llm`, `eval/` and `space/` as separate modules.

### Day 22: Offline index build
- **Built:** `python -m src.rag.build_index`: segments, embeddings, FAISS, metadata and cached Sentence-BERT embeddings. The app now starts in about 1.5 s.

### Day 23: RAG pipeline and LLM providers
- **Built:**
  - the reranker and the confidence score;
  - the context builder and the prompt;
  - Pydantic parsing;
  - citation validation, including removing sentences that only had fake citations;
  - follow-up rewriting;
  - Groq, Gemini, Ollama and mock clients with backoff.
- **Surprise:** "What is the capital of France?" was retrieved with high confidence. It turned out a Lecture 3 slide really shows that exact example prompt. Lesson: out-of-scope test questions must be *checked* against the transcripts, so the question script now does it automatically.

### Day 24: The Welsh bug
- **Found:** while drafting evaluation questions, Lecture 3's transcript turned out to be **Welsh gibberish**. Lectures 3, 4 and 6 had been auto-detected as Welsh (only 23–29% English-looking segments).
- **Fix:** force English; add a language sanity check; re-transcribe. Now **95% / 93%** English-looking for lectures 3 and 4. (Story 1.)

### Day 25: Evaluation
- **Built:** a 76-question set; `python -m eval.run`.
- **First result:** hybrid is best; **the reranker made things worse.**
- **Diagnosis:** the input order put noisy OCR text first. Transcript-first gave MRR **0.556 → 0.666**. Both rows are kept in the report, with a note that the fix was found on the same questions. (Story 2.)

### Day 26: API, demo, tests and docs (6 November)
- **Built:**
  - `POST /ask` and `GET /health`;
  - the Gradio demo with embedded players at the timestamp;
  - the Hugging Face Space packager with a privacy guard;
  - **25 pytest tests** with a mocked LLM;
  - README, the learning notes (later renamed DEVLOG) and a manual task list.
- Merged `feat/rag-layer` into `main`.

## Phase 5: Going live (11 – 12 November 2026)

### Day 27: Documentation and repository hygiene (11 November)
- **Rewrote the README** in a conventional GitHub layout: features, architecture diagram, setup, API reference, results, deployment, privacy.
- **Renamed** `LEARNING.md` to this `DEVLOG.md`, with the full walkthrough, build log and interview preparation.
- **Cleaned the commit history** so commits carry only the author's name, and force-pushed with `--force-with-lease` (which refuses if someone else pushed in the meantime). A backup branch was kept until the push was verified.

### Day 28: First real LLM run, and three bugs it exposed (11 November)
- **Added a Groq key** through a git-ignored `.env`.
- **Bug 1: the default model was retired.** Every request failed with `404 model_not_found` for `llama-3.1-8b-instant`.
  - **Fix:** listed the models the key can access; made `openai/gpt-oss-20b` the default; added automatic fallback to the first available model from a preference list.
  - Verified live, and covered by a unit test.
- **Bug 2: `.env` not found.** `load_dotenv()` searches from the *calling* file and failed when Python ran from stdin.
  - **Fix:** load it by absolute path from the project root.
- **Bug 3: over-eager follow-up detection.** "Why is softmax used…?" was glued to a previous question about pagination.
  - **Fix:** narrowed the regex to words that refer back; added tests for both directions.
- **Prompt fix:** answers cited only at the end and slipped in one outside fact. Rule 2 now demands a label at the end of every sentence, with an example.
- **The first full evaluation with Groq was distorted** by the free tier's 8,000 tokens-per-minute limit: a 23.7 s median "LLM latency" and 5 answers lost after retries. Rather than publish those numbers:
  - `post_json` now records the successful request's own time separately from waiting;
  - the evaluation pauses between questions (`--llm-pause`, default 45 s) and allows more retries;
  - citation accuracy counts only genuine `rag` answers, not error fallbacks;
  - LLM errors, rate-limited calls and judge errors are reported in the table.
- **Demo polish:** silenced the harmless Windows `ConnectionResetError` traceback; the latency line shows only real timings.
- **Lesson:** the demo running in the browser was started before the fix and kept failing. A long-running process keeps its old code, so restart after changing code.

### Day 29: Deployment (11 – 12 November)
- **Decision:** deploy on Hugging Face Spaces (free CPU), with the college's permission to publish the demo.
- **Constraint found:** the recordings can't be uploaded anywhere public (not YouTube, not the Space, and 4.5 GB is too big anyway).
- **Built:**
  - `prepare_space.py` now refreshes each segment's video URL from `data/video_meta.json` at packaging time, so adding a public link later needs no index rebuild;
  - it warns which lectures have no public video;
  - the "no video" citation message was reworded from an error-looking "Video file not available" to a clear note: the recordings aren't public, open the lecture at this timestamp.
- **Tested** the packaged folder on its own (port 7861): it loads the index, retrieves correctly, and falls back to retrieval-only with no key. 33 tests pass.
- **Wrote** the deployment guide (Part IV) and the run guide (Part II).

---

# Part VII: Results

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

**Being re-measured.** The first run with Groq (`openai/gpt-oss-20b`) was distorted by the free tier's rate limit (Day 28), so its numbers are not reported. The paced re-run writes faithfulness, citation accuracy, end-to-end refusal, model latency (`llm_request_ms`) and error counts to `reports/eval_results.md`; this section will quote them once that run is finished.

**One observed example** (from the demo, not a metric): *"Why is softmax used at the output layer?"* was answered in `rag` mode with citations to Lecture 1, 48:33 and Lecture 5, 19:03. The model took 1.5 s and the whole request 3.4 s, about 1.9 s of it in the reranker.

### Reading the results
1. **Hybrid beats either retriever alone.** BM25 and dense fail on different questions.
2. **BM25 beats dense here.** Lecture questions are full of exact technical terms ("perplexity", "bigram", "LSTM"), and the drafted questions reuse lecture wording.
3. **The reranker is a mixed result.** Fixing its input made it best on the full set, but on the 17 human-written questions it is worse than hybrid and adds about 1.8 s. It stays switchable until more human questions decide.
4. **The fusion encoder is the weakest part** (MRR 0.20). It needs retraining on real questions.
5. **The refusal gate works on these questions**, but the held-out set has only a handful of out-of-scope questions, so it's a weak estimate.
6. **Latency is dominated by the reranker and the LLM**, not by search: retrieval takes milliseconds, the reranker about 1.8 s on CPU, and the LLM about 1.5 s.

---

# Part VIII: Lessons, limitations and next steps

### Lessons
1. **Silent failures are the most dangerous bugs.** Empty OCR, random-vector fallbacks, repeated transcripts and Welsh decoding all *ran without errors*. Validate data, not just code, and make pipelines fail loudly.
2. **Baselines first.** BM25 is hard to beat; every learned component must earn its place in an ablation.
3. **Don't evaluate on what you trained on.** Pseudo-queries come from the segment's own text, so finding them is trivial.
4. **Report the bad numbers too.** The original reranker row and the weak fusion encoder stay in the tables.
5. **Input construction is part of the model.** Field order and truncation changed reranker MRR by 0.11.
6. **Measure what you think you're measuring.** "LLM latency" was mostly rate-limit waiting until the request time was recorded separately.
7. **External services change under you.** A vendor retired the model overnight; the client now adapts instead of breaking.
8. **Design for the deployment early.** Because heavy work was offline, deploying meant shipping 9.4 MB, and losing the videos cost the demo nothing essential.

### Limitations
- One institution, one course, one camera; 5 lectures indexed; small evaluation set with wide confidence intervals.
- 49 of the 66 in-scope questions are drafted from transcripts and not yet human-verified (they favour lexical retrieval).
- No human boundary annotations yet, so no segmentation scores.
- The fusion encoder was trained on pseudo-queries.
- Generation metrics are still being measured.
- LLM-as-judge limitations: leniency towards fluent text, shared biases with the generator, inconsistent claim splitting.
- The deployed demo has no video playback (the recordings can't be published).
- The free Groq tier limits how many people can use the demo at once.
- Privacy: transcripts in the deployed index contain anything said in class, including by students.

### Next steps
1. Verify the drafted questions; annotate boundaries; add a second annotator for agreement.
2. Finish the generation evaluation and report it.
3. Retrain Stage 2 on human questions; re-run the fusion ablation (cross-attention vs concat vs pairwise).
4. Index all lectures (about 30 planned).
5. Slide snapshots for the deployed demo (cropped projector screen at each segment: no audio, no faces).
6. A small user study: how long does a student need to find an answer with and without ConceptBound?

---

# Part IX: Interview questions and answers

### Project overview
1. **Explain the project in 30 seconds.** ConceptBound lets students ask questions about recorded lectures. It cuts each lecture into concept segments with a trained boundary detector, indexes them with transcript, slide text and visual features, retrieves the right moments with hybrid BM25 plus dense search, and has an LLM answer **only from those moments**, with `[Lecture, mm:ss]` citations. It refuses when the lectures don't cover the question. It's deployed as a free Hugging Face Space.
2. **What was your role / the hardest part?** Building the end-to-end pipeline and, hardest of all, finding the silent data failures (empty OCR, Welsh transcripts, repeated text) that made the numbers wrong without any error message.
3. **Why two stages?** Stage 1 decides *where* concepts start and end; Stage 2 decides *what* each segment is about. Concept-sized units give better retrieval targets and cleaner LLM context than arbitrary windows.
4. **What would you do differently from the start?** Collect 50 human questions in week one. Almost every wrong turn (pseudo-query evaluation, the reranker regression found late) came from not having real questions early.

### Data and features
5. **Why extract features every 3 seconds?** It's fine enough to place boundaries within seconds, and small enough (about 1,300 steps per hour) for cheap sequence modelling.
6. **How did you get clean slide text from a wide classroom shot?** I detected the projector screen automatically: the median frame removes the moving instructor, Otsu thresholding finds the bright area, and I keep the largest screen-shaped region per 5-minute chunk. Then I OCR only that crop, and only when the screen changes (SSIM).
7. **Why the median and not the mean of the frames?** The mean blends the instructor into a ghost that still darkens the screen; the median ignores anything present in fewer than half the frames, so a moving person disappears completely.
8. **What is topic drift?** The cosine distance between Sentence-BERT embeddings of consecutive 30-second transcript windows. It's high when the spoken topic changes.
9. **How do you handle Whisper hallucinations?** Confidence filters (`no_speech_prob`, `avg_logprob`, compression ratio), no conditioning on previous text, and removal of short phrases repeated 3+ times in 60 s.
10. **Why chunk-and-average long texts instead of truncating?** Sentence-BERT reads 256 tokens. Truncating a 5-minute transcript represents it by its first 20 seconds; averaging chunk embeddings keeps information from the whole segment.

### Models
11. **How are positive and negative pairs formed in Stage 1?** Same-segment neighbours are positives and boundary-crossing neighbours are negatives (margin loss). For InfoNCE, each step is an anchor with one same-segment positive and four other-segment negatives.
12. **What does the margin do?** It only penalizes boundary-crossing neighbours while their similarity is above 0.5, so the model doesn't waste capacity pushing them further apart.
13. **Why an embedding-based boundary score rather than a classifier?** Boundaries are rare (class imbalance); a similarity score is smooth, comparable across lectures and naturally peak-pickable.
14. **Why an adaptive threshold for peaks?** The absolute scale of `1 − cos` depends on the embedding size and training length; `mean + std` of each lecture's own scores transfers across checkpoints.
15. **Explain the Stage 2 fusion encoder.** Three modality embeddings become three tokens with type embeddings; one self-attention block lets them attend to each other; pooled and normalized into one segment vector. A query projector maps questions into the same space.
16. **Why modality-type embeddings?** Self-attention is permutation-invariant; without them the model can't tell which token is the transcript and which is the slide.
17. **What are in-batch negatives?** In a batch of B pairs, each query's other B−1 segments act as negatives, so no extra negative mining is needed.
18. **Why mask same-segment pairs in the loss?** Two questions can target the same segment; treating it as a negative would push identical items apart (a false negative).
19. **What does temperature do in InfoNCE?** It divides the similarities before the softmax. A low τ (0.07) sharpens the distribution, so the loss focuses on the hardest negatives; too low makes training unstable.
20. **What's the pairwise fusion variant?** It's the base paper's idea: unimodal experts, one shared block fusing each modality pair, plus a loss per pair. It's an ablation against the default.

### Retrieval
21. **Why FAISS `IndexFlatIP` and not HNSW or IVF?** It's exact, and inner product equals cosine on normalized vectors. With hundreds of segments it's sub-millisecond; approximate indexes add recall loss and tuning for no gain at this scale.
22. **BM25 vs dense retrieval?** BM25 is exact-term matching: precise on jargon, untrained, explainable, but blind to synonyms. Dense matches meaning but can drift and is weaker on rare tokens. Here BM25 actually wins (0.613 vs 0.549 MRR).
23. **How is hybrid computed, and why normalize?** Top candidates from both; BM25 divided by its max to fall in 0–1, because raw BM25 is unbounded and would dominate the cosine; averaged 50/50. Reciprocal rank fusion is a scale-free alternative, used in the research harness.
24. **Why does a reranker help in general?** Bi-encoders embed the query and passage separately; a cross-encoder reads them together, so it matches specific words across both. It's too slow for the whole corpus, so it only re-scores the top 20.
25. **Did the reranker help here?** It depends on the input: OCR-first made it worse (MRR 0.556); transcript-first made it best overall (0.666). But on human-written questions it's below hybrid, so it's switchable, not trusted blindly.
26. **What is the variance trigger?** If the fusion model's scores are all low, all high or nearly identical, the model is untrained or collapsed, so `auto` mode falls back to hybrid. It keeps the app working whatever the model's state.

### RAG and LLM
27. **How does RAG reduce hallucination?** The LLM answers from retrieved evidence and is told to say "not covered" otherwise; a confidence gate blocks off-topic questions before the LLM; and citations are validated. It reduces hallucination; it doesn't make it impossible.
28. **What is the refusal gate and how is its threshold chosen?** If retrieval confidence (reranker probability or best cosine) is below the threshold, the system refuses without calling the LLM. The threshold maximizes balanced accuracy on dev questions and is reported on held-out ones.
29. **What happens if the LLM cites a segment that wasn't retrieved?** The label is dropped, sentences that relied only on it are removed, and if no valid citation remains the answer becomes a refusal.
30. **Where do the timestamps come from? Can they be hallucinated?** No. The LLM only writes `[S1]`; the code maps S1 to its segment and reads `start_time` from the index. A timestamp is as accurate as the segment boundary.
31. **Why `[S1]`-style labels instead of real segment IDs in the prompt?** LLMs copy short labels reliably; long IDs get mangled. The code maps them back to the exact IDs.
32. **How do you handle malformed JSON from the LLM?** A tolerant parser strips fences and tries `raw_decode` from every `{`, then Pydantic validates and coerces types. On failure the API returns the top moments with `mode="llm_error"`; it never crashes.
33. **How are follow-up questions handled?** Questions of 4 words or fewer, or ones that refer back ("it", "that", "explain more"), get the previous two user questions prepended before retrieval. It's deterministic and testable. I had to narrow it after "Why is…" questions were wrongly treated as follow-ups.
34. **How do you manage the token budget?** At most 1,800 tokens of context, about 450 per segment, estimated as words × 1.3; overlapping segments are removed first, lowest-ranked trimmed first.
35. **Why temperature 0?** The answer should be the most likely, repeatable reading of the evidence; creativity is a liability in a grounded QA system.
36. **How do you switch providers and handle rate limits?** An `LLMClient` interface chosen by `LLM_PROVIDER`. Retries with exponential backoff and jitter only for transient errors (429, 5xx, timeouts), `Retry-After` support, and clear error types. The app degrades to showing the moments.
37. **Why not retry a 401?** A wrong key won't become right by waiting; retrying only delays a clear error message.
38. **What happened when Groq retired your model?** Every request returned 404. I made the client list the available models and switch to the first one from a preference list, only on that specific error, and covered it with a test.

### Evaluation
39. **How did you evaluate retrieval?** Recall@5/10, MRR and nDCG@10 against gold segments, with bootstrap confidence intervals, as an ablation of BM25, dense, fusion, hybrid, auto and reranked hybrid. Human-written questions are reported separately.
40. **Why store gold answers as time spans, not segment IDs?** IDs change every time lectures are re-segmented; time spans don't. The evaluation maps spans to the current segments with a two-sided overlap rule.
41. **Why are pseudo-queries never used for evaluation?** They're built from the segment's own text, so lexical search finds them trivially and the score means nothing.
42. **Why balanced accuracy for the refusal gate?** There are far more in-scope questions than out-of-scope ones, so "never refuse" would look good on plain accuracy.
43. **What is faithfulness, and what are the limits of LLM-as-judge?** The share of answer claims supported by the cited segments, judged by an LLM with a fixed rubric that sees only those segments. Limits: leniency to fluent text, shared biases with the generator, inconsistent claim splitting, and cost. Spot-check by hand.
44. **How do you know the out-of-scope questions are really out of scope?** The question script searches the index for each topic's key terms and warns if any appear. That caught "capital of France", which a slide really contains.
45. **Your first LLM latency was 23.7 s. Was the model slow?** No: that was mostly waiting on the free tier's tokens-per-minute limit. I timed the successful request separately (`llm_request_ms`), paced the evaluation, and report rate-limit retries as their own number.

### Engineering and deployment
46. **Why build the index offline?** Whisper and OCR take minutes per lecture on CPU; the app loads precomputed files and starts in about 1.5 s on a free CPU.
47. **How did you test an LLM app without paying for API calls?** A mock `LLMClient` with scripted outputs; 33 pytest tests cover citation validation, refusal, malformed output, retries, the retired-model fallback, follow-up detection and the factory.
48. **How did you deploy it, and why there?** Hugging Face Spaces, free CPU tier: enough RAM for the small models, native Gradio, encrypted secrets and git-based deploys. A packaging script copies only inference code, the 0.9 MB index and the 8 MB checkpoint.
49. **How do you keep the API key safe in a public repo?** It's never in a file that's committed: locally in a git-ignored `.env`, on the Space as an encrypted secret injected as an environment variable. The packager's allow-list can't copy `.env`.
50. **The videos couldn't be published. Doesn't that break the demo?** No. Answering needs only the transcripts, slide text and timestamps, which are in the index. The deployed demo shows the timestamp and quote; the local version still plays the video.
51. **How did you handle privacy?** Videos, transcripts and indexes are git-ignored; `lecture_span` removes the roll call; the packager refuses internal lectures unless explicitly allowed (we had permission); no video is published.
52. **Why CPU-only PyTorch in the Space requirements?** The default wheel bundles CUDA (~2 GB); the CPU wheel is ~200 MB, which makes builds faster and fits the free tier.
53. **How would you scale to 10,000 lectures?** An approximate FAISS index (HNSW or IVF-PQ), a search engine for BM25, batch or GPU extraction, a smaller or GPU reranker, caching of frequent questions, async LLM calls, a paid LLM tier, and a larger human-labelled evaluation set.
54. **What would you do with one more month?** A verified 100+ question evaluation set, boundary annotations, retraining Stage 2 on human questions, generation metrics, and a user study on time-to-answer.

### Code-reading questions (they may show you a snippet)
55. **In `adjacent_margin_loss`, why `F.relu(sim - (1 - margin))`?** It's zero once similarity is below `1 − margin`, so already-separated pairs contribute no gradient.
56. **In `info_nce_loss`, why `labels = torch.arange(B)`?** Query *i*'s correct segment is segment *i*, so the right class for row *i* is the diagonal entry.
57. **In `validate_citations`, what if the LLM writes `[S1, S7]` and S7 doesn't exist?** S7 is dropped, S1 is kept and rendered as `[Lecture, mm:ss]`; the sentence stays because it still has one valid source.
58. **In `post_json`, why is the backoff `backoff * 2 ** attempt + random.uniform(0, 0.5)`?** Doubling waits gives an overloaded server more room each time; the random jitter stops many clients from retrying in sync.
59. **In `SearchService.search`, why retrieve 20 candidates when the user wants 5?** The reranker can only reorder what it's given; 20 gives it room to promote a good segment the first stage ranked 12th.
60. **In `build_context`, why `break` when the budget drops below 40 tokens?** A segment with fewer than ~30 words of context is more likely to mislead than help; better to stop.

### Behavioural
61. **Tell me about a time your results were wrong.** See Story 1 (silent data failures).
62. **Tell me about a time an "improvement" made things worse.** See Story 2 (the reranker).
63. **How do you make sure you're not fooling yourself?** Human gold data only, baselines in every table, dev/test separation for thresholds, confidence intervals, and reporting the bad numbers as well, including the rate-limited evaluation I chose not to publish.
64. **Tell me about working under a constraint you couldn't change.** The recordings couldn't be published. I separated what the demo *needs* (text and timestamps, in a 1 MB index) from what was *nice* (video), shipped text-only citations, and kept video in the local version.

---

# Part X: Two interview stories

## Story 1: "Tell me about a difficult bug you found and how you solved it."

**Situation.** My lecture search system was producing numbers and running without a single error. While preparing evaluation questions I read the transcript of Lecture 3 and found it was gibberish: *"Yn yw'n gweithio'r ddod…"*, which is Welsh, for an English lecture.

**Task.** I needed to find out how far the corruption went, fix it at the root, and make sure the same class of bug couldn't hide again, because every downstream number (topic drift, retrieval, the evaluation) depended on those transcripts.

**Action.**
1. **Measured the damage.** I wrote a quick check counting how many transcript segments contain common English function words. Lectures 3, 4 and 6 scored only 23–29%; the others were around 90%.
2. **Found the root cause.** Whisper auto-detects the language from the **first 30 seconds** of audio, and a classroom recording starts with room noise and chatter. So it guessed Welsh and decoded the whole hour as Welsh.
3. **Fixed it at the source.** I forced `language="en"` in the configuration and re-transcribed only the three affected lectures.
4. **Prevented recurrence.** I turned the quick check into `english_share()`, which prints a loud warning when less than half of a transcript looks like English.
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
# Install
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
copy .env.example .env                                # then add GROQ_API_KEY

# Demo and API
python space/app.py                                   # Gradio on http://127.0.0.1:7860
uvicorn app.backend.main:app --port 8000              # API, docs at /docs

# Build / rebuild the RAG index (no Whisper/OCR unless --extract-missing)
python -m src.rag.build_index --ids college_lec01 college_lec02 college_lec03 college_lec04 college_lec05

# Evaluate the RAG layer
python -m eval.make_questions && python -m eval.run   # -> reports/eval_results.md
python -m eval.run --no-generation                    # retrieval + refusal only, no API calls

# Tests
python -m pytest tests -q

# Deploy
python space/prepare_space.py --allow-internal-videos --license other   # -> build/hf_space/
cd build/hf_space && git add . && git commit -m "Update" && git push     # after the first push

# Research pipeline
python scripts/extract_features.py                    # Whisper + OCR + CLIP (slow)
python scripts/refresh_features.py --ocr --asr        # re-run OCR/ASR on existing features
python scripts/run_paper_pipeline.py                  # train, segment, evaluate, ablate, rebuild paper
```

| Path | Purpose |
|---|---|
| `config.py` | Every path, hyperparameter and model name |
| `src/feature_extraction/` | Whisper, OCR with screen detection, CLIP, optical flow |
| `src/stage1_boundary/` | Boundary detector: model, losses, pseudo-labels, training, segmentation |
| `src/stage2_retrieval/` | Fusion encoder, training, FAISS index, original hybrid retriever |
| `src/retrieval/` | Search service, reranker, confidence |
| `src/rag/` | Index build, context, prompts, schema, citations, rewrite, pipeline |
| `src/llm/` | Provider interface, HTTP retries, Groq/Gemini/Ollama/mock clients, factory |
| `eval/` | RAG evaluation (questions, gold mapping, metrics, judge, runner) |
| `scripts/` | Research pipeline, annotation tools, ablations, measurement |
| `app/` | FastAPI backend and HTML frontend |
| `space/` | Gradio demo, pinned Space requirements, Hugging Face packager |
| `tests/` | pytest suite (33 tests, mocked LLM) |
| `paper/` | ICALT 2027 paper draft generator and figures |
| `data/video_meta.json` | Per-lecture title, URL, licence, attribution |
| `docs/ANNOTATION_GUIDELINES.md` | How to annotate boundaries and queries |
| `MANUAL_TASKS.md` | What still needs a human |
