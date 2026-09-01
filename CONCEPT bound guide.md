# 🎓 Lecture Video Semantic Search --- Interview Revision README

## 1. 🎯 What does the project do?

**Goal:** Given a long lecture video and a natural-language query, find
the **most relevant concept/segment** and return its **video +
timestamp**.

Example:

> Query: "Explain backpropagation with an example."

Output:

> Lecture 3 --- 24:10--26:45 → relevant explanation + **Jump to moment**

### Core idea

**Video → concept-level segments → multimodal embeddings → semantic
retrieval**

------------------------------------------------------------------------

# 2. 🧠 Overall Architecture

The system has **2 trainable stages**:

``` text
             Lecture Video
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
      OCR        ASR       CLIP
    Slides      Whisper     Vision
       │          │          │
       └──────────┼──────────┘
                  ↓
         Temporal Features
                  ↓
       ┌────────────────────┐
       │      STAGE 1       │
       │ Boundary Detector  │
       └────────────────────┘
                  ↓
        Concept-Coherent
             Segments
                  ↓
       ┌────────────────────┐
       │      STAGE 2       │
       │ Cross-Modal Fusion │
       └────────────────────┘
                  ↓
          Segment Embeddings
                  ↓
               FAISS
                  ↑
                Query
                  ↓
          Ranked Segments
                  ↓
          Timestamp / Jump
```

### Why two stages?

-   **Stage 1:** *Where does one concept end and another begin?*
-   **Stage 2:** *Which concept is relevant to my query?*

------------------------------------------------------------------------

# 3. 📹 Feature Extraction

The video provides **three main modalities**:

  Modality          Technology    Purpose
  ----------------- ------------- -----------------------
  📝 Board/Slides   **EasyOCR**   Extract visible text
  🎙️ Speech         **Whisper**   Generate transcript
  🖼️ Visual         **CLIP**      Encode visual content

### Additional signal

**Sentence-BERT** is applied to rolling transcript windows to calculate
**cosine distance / topic drift**.

So Stage 1 receives signals such as:

``` text
OCR change
+ Topic drift
+ Visual change
        ↓
Temporal feature sequence
```

------------------------------------------------------------------------

# 4. 🔵 Stage 1 --- Pedagogical Boundary Detector

### Objective

Find **concept boundaries**, rather than arbitrary time intervals.

Example:

``` text
00:00 ───────── 05:20
       Gradient Descent

05:20 ───────── 09:10
       Backpropagation

09:10 ───────── 13:30
       Worked Example
```

## Why not simple classification?

Instead of:

``` text
boundary = 0 / 1
```

the project learns **embeddings for each time-step**.

Desired behavior:

``` text
Same concept:
z₁ ≈ z₂ ≈ z₃

Boundary:
z₃ <<<different>>> z₄
```

### Boundary score

``` text
bₜ = 1 − cosine(zₜ, zₜ₊₁)
```

Higher score → more likely boundary.

------------------------------------------------------------------------

# 5. 🧩 Stage 1 Models

Three interchangeable encoders were implemented:

### CNN

-   Dilated 1D convolutions
-   Fast
-   Good local context

### Transformer

-   Self-attention
-   Captures long-range dependencies

### BiLSTM

-   Bidirectional temporal context
-   Supports variable-length sequences

------------------------------------------------------------------------

# 6. 🏋️ Stage 1 Training

Two losses are combined.

### Margin Loss

``` text
Same segment → embeddings closer
Different segments → embeddings farther apart
```

### Segment InfoNCE

For an anchor:

``` text
Positive = another timestep in same segment
Negatives = timesteps from other segments
```

### Combined objective

``` text
Total Loss =
Margin Loss + 0.5 × InfoNCE
```

### Important trick: pseudo-labels

When manually labelled boundaries are limited, peaks in:

-   OCR change
-   Topic drift
-   Visual change

are used as **pseudo-boundaries** for warm-start training.

Then the model can be fine-tuned using manual annotations.

------------------------------------------------------------------------

# 7. 🔎 Stage 2 --- Cross-Modal Retrieval

After segmentation, every segment has:

``` text
OCR embedding
ASR embedding
Visual embedding
```

These are treated as **3 tokens**:

``` text
[ OCR ] [ ASR ] [ Visual ]
```

A Transformer performs **self-attention across the modalities**.

``` text
3 modality tokens
       ↓
Transformer
       ↓
Mean Pooling
       ↓
L2 Normalization
       ↓
z_segment
```

### Why attention?

It allows the model to learn which modality/interactions are useful
instead of relying on fixed weights.

------------------------------------------------------------------------

# 8. ❓ Query Processing

Example:

> "Explain backpropagation with an example"

Processing:

``` text
Query
  ↓
Sentence-BERT
  ↓
Query Projector
  ↓
z_query
```

The query and segment are projected into the **same embedding space**.

Therefore:

``` text
similarity(z_query, z_segment)
```

measures relevance.

------------------------------------------------------------------------

# 9. 🏋️ Stage 2 Training

Uses **symmetric InfoNCE**.

For a batch:

``` text
Query A → Correct Segment A ✅
Query A → Segment B ❌
Query A → Segment C ❌
```

And simultaneously:

``` text
Segment A → Query A ✅
Segment A → Query B ❌
Segment A → Query C ❌
```

This trains matching query-segment pairs to have high similarity.

**Optimizer:** AdamW

------------------------------------------------------------------------

# 10. ⚡ FAISS Retrieval

After training:

``` text
All segment embeddings
        ↓
      FAISS
```

Use:

``` text
IndexFlatIP
```

Because embeddings are **L2-normalized**, inner product is equivalent to
cosine similarity.

At query time:

``` text
Query
 ↓
Query embedding
 ↓
FAISS
 ↓
Top-K segments
```

------------------------------------------------------------------------

# 11. 🛟 Why BM25 Fallback?

A neural retrieval model can fail or **collapse during early training**,
especially with insufficient labelled query/segment pairs.

So the system also maintains:

-   **BM25** → strong keyword/lexical matching
-   **Sentence-BERT similarity** → semantic text matching

If FAISS scores look unreliable:

``` text
Neural Retrieval
      ↓
Score distribution check
      ↓
Unreliable?
      ↓
BM25 + Sentence-BERT fallback
```

This improves robustness.

------------------------------------------------------------------------

# 12. 🌐 Application Layer

### Backend

**FastAPI**

``` text
GET /search?q=...&top_k=5
```

### Frontend

Simple interface showing:

-   Video ID
-   Timestamp
-   Text snippet
-   Jump-to-moment link

------------------------------------------------------------------------

# 13. 📊 Evaluation

## Stage 1 --- Segmentation

Compare:

``` text
Our Boundary Detector
        vs
Fixed Windows
        vs
Shot Detection
```

### Metrics

**Boundary F1**

Checks whether predicted boundaries match ground truth within a
tolerance, e.g. **±15 seconds**.

**Mean IoU**

Measures overlap between predicted and ground-truth segments.

------------------------------------------------------------------------

## Stage 2 --- Retrieval

### Recall@K

``` text
Recall@1
Recall@5
Recall@10
```

Question:

> Did the correct segment appear in the top K results?

### Mean IoU@1

Measures overlap between the **top-1 retrieved segment** and the
ground-truth segment.

------------------------------------------------------------------------

# 14. 🧪 Important Ablation Studies

### Ablation 1 --- Remove signals

``` text
All signals
vs
No OCR
vs
No Topic Drift
vs
No Visual
```

**Purpose:** Determine which signal contributes most to boundary
detection.

------------------------------------------------------------------------

### Ablation 2 --- Segmentation

``` text
Stage 1 segments
vs
Fixed windows
```

**Purpose:** Test whether better segmentation improves retrieval.

------------------------------------------------------------------------

### Ablation 3 --- Fusion

``` text
Cross-Attention
vs
Concatenation
```

**Purpose:** Test whether learned cross-modal interaction improves
retrieval.

------------------------------------------------------------------------

# 15. 🔥 Most Important Technical Points

If you only have **5 minutes before the interview**, remember these:

### 1. Two-stage architecture

> **Stage 1 finds concept boundaries; Stage 2 retrieves relevant
> concepts.**

### 2. Three modalities

> **OCR + Whisper/ASR + CLIP**

### 3. Stage 1 representation

> Learn embeddings where same-concept timesteps are close and
> boundary-crossing timesteps are far apart.

### 4. Stage 1 loss

> **Margin Loss + 0.5 × Segment InfoNCE**

### 5. Boundary score

``` text
1 − cosine(zₜ, zₜ₊₁)
```

### 6. Stage 2 fusion

> Treat OCR, ASR and visual embeddings as **3 tokens** and use
> Transformer self-attention.

### 7. Stage 2 training

> **Symmetric InfoNCE** on query-segment pairs.

### 8. Retrieval

> **FAISS + cosine similarity**

### 9. Robustness

> **BM25 + Sentence-BERT fallback** when neural retrieval looks
> unreliable.

### 10. Evaluation

> Stage 1 → **Boundary F1 + Mean IoU**

> Stage 2 → **Recall@1/5/10 + Mean IoU@1**

------------------------------------------------------------------------

# 🎤 30-Second Interview Explanation

> **"My project is a two-stage semantic search system for long lecture
> videos. In Stage 1, I use OCR, Whisper transcripts, CLIP visual
> features and transcript topic-drift signals to detect pedagogically
> meaningful concept boundaries. Instead of treating boundary detection
> as simple binary classification, I learn temporal embeddings using a
> combination of margin loss and Segment InfoNCE, where embeddings
> within the same concept are pulled together and boundary-crossing
> embeddings are separated.**
>
> **In Stage 2, each detected segment has OCR, ASR and visual
> embeddings. I treat these as three modality tokens and use Transformer
> self-attention to fuse them into a single segment embedding. A query
> is encoded using Sentence-BERT and projected into the same space.
> Symmetric InfoNCE trains matching query-segment pairs to have high
> similarity. The resulting embeddings are indexed using FAISS for fast
> retrieval, with BM25 and Sentence-BERT as a fallback when the neural
> model is unreliable. Finally, I evaluate segmentation using Boundary
> F1 and IoU, and retrieval using Recall@K and Mean IoU@1."**

------------------------------------------------------------------------

# 🧠 One-Line Mental Model

``` text
Video → Extract 3 modalities → Find concepts → Fuse modalities → Embed → FAISS → Retrieve timestamp
```

## Technology Stack

  Layer                        Technology
  ---------------------------- ---------------
  OCR                          EasyOCR
  ASR                          Whisper
  Text Embeddings              Sentence-BERT
  Vision                       CLIP
  Deep Learning                PyTorch
  Vector Search                FAISS
  Lexical Search               BM25
  Backend                      FastAPI
  Computer Vision              OpenCV
  Evaluation / preprocessing   scikit-learn
