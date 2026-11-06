"""
ASR transcription (Whisper) + topic-drift signal.

Pipeline:
  1. Transcribe the full lecture audio with Whisper, obtaining a list of
     (start, end, text) segments with word/segment-level timestamps.
  2. For each fixed time-step of the multimodal feature stream, gather the
     transcript text spoken in a rolling window centred on that time-step.
  3. Encode each rolling-window transcript with Sentence-BERT.
  4. The "topic-drift" signal at time t is the embedding distance between the
     rolling-window transcript ending at t and the one ending at t-1 - large
     values indicate the spoken content is changing topic.

Both Whisper and Sentence-BERT are *frozen, pretrained* models used purely
for feature extraction (no gradients, no fine-tuning).

Whisper hallucinates on silence and room noise (e.g. "I'm going to go to the
next room" repeated before a lecture starts). Segments that look like
hallucinations are dropped using Whisper's own confidence statistics, see
`is_probable_hallucination`.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class WhisperASR:
    """Thin wrapper around openai-whisper for offline transcription."""

    def __init__(self, model_name: str = "base", device: str = None):
        self._model = None
        self.model_name = model_name
        self.device = device

    def _load(self):
        if self._model is None:
            import whisper  # lazy import
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        model = self._load()
        # condition_on_previous_text=False stops one hallucination from
        # seeding the next window (the main source of repeated phrases).
        from config import MODELS
        result = model.transcribe(audio_path, verbose=False, language=MODELS.whisper_language or None,
                                  condition_on_previous_text=False)
        kept, dropped = [], 0
        for seg in result["segments"]:
            if is_probable_hallucination(seg):
                dropped += 1
                continue
            kept.append(TranscriptSegment(start=seg["start"], end=seg["end"],
                                          text=seg["text"].strip()))
        kept = drop_repeated_segments(kept)
        share = english_share([s.text for s in kept])
        if share < 0.5:
            print(f"    WARNING: only {100 * share:.0f}% of transcript segments look like English. "
                  f"Check the language setting (config.MODELS.whisper_language).")
        print(f"    Whisper: kept {len(kept)} segments, dropped {dropped} "
              f"low-confidence + {len(result['segments']) - dropped - len(kept)} repeated")
        return kept


def is_probable_hallucination(seg: dict, no_speech_thresh: float = 0.6,
                              logprob_thresh: float = -1.0,
                              compression_thresh: float = 2.4) -> bool:
    """Whisper's recommended filters: silence that was transcribed anyway
    (high no_speech_prob with low confidence) or degenerate repetitive text
    (high gzip compression ratio)."""
    text = seg.get("text", "").strip()
    if not text:
        return True
    if seg.get("compression_ratio", 0.0) > compression_thresh:
        return True
    if seg.get("no_speech_prob", 0.0) > no_speech_thresh and seg.get("avg_logprob", 0.0) < logprob_thresh:
        return True
    return False


def drop_repeated_segments(segments: List[TranscriptSegment], window_sec: float = 60.0,
                           min_count: int = 3, max_words: int = 6) -> List[TranscriptSegment]:
    """Drop looped hallucinations: a short phrase (<= `max_words` words) that
    occurs `min_count` or more times within `window_sec` seconds, e.g.
    "I don't know. / No, no. / I don't know." over room noise. Real lecture
    speech rarely repeats a short sentence verbatim that often."""
    def norm(t):
        return " ".join(t.lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split())

    keys = [norm(s.text) for s in segments]
    drop = set()
    for i, seg in enumerate(segments):
        if len(keys[i].split()) > max_words:
            continue
        same = [j for j in range(len(segments))
                if keys[j] == keys[i] and abs(segments[j].start - seg.start) <= window_sec]
        if len(same) >= min_count:
            drop.update(same)
    return [s for i, s in enumerate(segments) if i not in drop]


def english_share(texts) -> float:
    """Fraction of segments whose words are at least 15% common English
    function words: a cheap sanity check for wrong-language decoding."""
    import re
    common = set("the a is and to of in that this we you it so what are for be on with have can do not if one will".split())
    n = hits = 0
    for t in texts:
        w = re.findall(r"[a-z]+", t.lower())
        if w:
            n += 1
            hits += sum(x in common for x in w) / len(w) > 0.15
    return hits / n if n else 0.0


def asr_segments_to_array(transcript: List[TranscriptSegment]) -> np.ndarray:
    """(N, 3) object array of [start, end, text] for saving in the .npz."""
    arr = np.empty((len(transcript), 3), dtype=object)
    for i, seg in enumerate(transcript):
        arr[i] = [float(seg.start), float(seg.end), seg.text]
    return arr


def asr_segments_from_array(arr) -> List[TranscriptSegment]:
    return [TranscriptSegment(float(a[0]), float(a[1]), str(a[2])) for a in arr]


class DummyASR:
    """Fallback used when Whisper / ffmpeg is unavailable. Produces an empty
    transcript so the pipeline can still run end-to-end."""

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        return []


def get_asr_backend(model_name: str = "base", use_dummy: bool = False, device: str = None):
    """Whisper backend. `use_dummy=True` gives an empty transcript and is for
    smoke tests only; real runs raise if Whisper cannot load."""
    if use_dummy:
        return DummyASR()
    return WhisperASR(model_name, device=device)


class SentenceEncoder:
    """Wrapper around a frozen Sentence-BERT model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self._model = None
        self.model_name = model_name
        self._device = device
        self._dim = 384  # all-MiniLM-L6-v2 default

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            self._dim = self._model.get_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        # No random-vector fallback: if the model cannot load, fail loudly
        # rather than produce embeddings that look valid but mean nothing.
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        model = self._load()
        embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeds.astype(np.float32)

    def encode_long(self, texts: List[str], chunk_words: int = 150) -> np.ndarray:
        """Encode texts longer than the model's 256-token limit by averaging
        the embeddings of ~`chunk_words`-word chunks (plain `encode` would
        silently truncate a 5-minute segment to its first few sentences)."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        chunks, owner = [], []
        for i, text in enumerate(texts):
            words = str(text).split()
            parts = [" ".join(words[j:j + chunk_words]) for j in range(0, len(words), chunk_words)] or [""]
            chunks.extend(parts)
            owner.extend([i] * len(parts))
        embeds = self.encode(chunks)
        embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
        owner = np.array(owner)
        out = np.stack([embeds[owner == i].mean(axis=0) for i in range(len(texts))])
        return out.astype(np.float32)


def text_for_window(transcript: List[TranscriptSegment], t_center: float, half_window: float) -> str:
    """Concatenate transcript text whose interval overlaps
    [t_center - half_window, t_center + half_window]."""
    lo, hi = t_center - half_window, t_center + half_window
    parts = [seg.text for seg in transcript if seg.end >= lo and seg.start <= hi]
    return " ".join(parts).strip()


def texttiling_prefilter(window_texts: List[str],
                         tfidf_threshold: float = 0.3,
                         margin: int = 2) -> np.ndarray:
    """Lightweight TF-IDF-based pre-filter to identify candidate topic shifts.

    Computes TF-IDF vectors for each rolling-window text and flags positions
    where the cosine distance between consecutive windows exceeds the threshold.
    A margin of ±`margin` steps is added around each candidate to ensure
    nearby context is also encoded with Sentence-BERT.

    Returns a boolean mask (T,) where True = should be encoded by SBERT.
    """
    T = len(window_texts)
    if T <= 2:
        return np.ones(T, dtype=bool)

    # Filter out empty texts to avoid TF-IDF issues
    non_empty = [t if t.strip() else "." for t in window_texts]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_distances

        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vectorizer.fit_transform(non_empty)

        # Cosine distance between consecutive windows
        distances = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            distances[t] = float(cosine_distances(tfidf[t-1:t], tfidf[t:t+1])[0, 0])

        # Flag candidates where distance exceeds threshold
        candidates = distances > tfidf_threshold
    except Exception:
        # If sklearn unavailable, fall back to encoding everything
        return np.ones(T, dtype=bool)

    # Expand candidates by ±margin steps for context
    mask = np.zeros(T, dtype=bool)
    mask[0] = True  # always include first
    mask[-1] = True  # always include last
    for t in range(T):
        if candidates[t]:
            lo = max(0, t - margin)
            hi = min(T, t + margin + 1)
            mask[lo:hi] = True

    # Also sample every 10th step for baseline coverage
    mask[::10] = True

    return mask


def extract_topic_drift_signal(
    transcript: List[TranscriptSegment],
    timestamps: np.ndarray,
    encoder: SentenceEncoder,
    half_window: float = 15.0,
    use_hybrid: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Compute topic-drift signal over the given timestamps.

    Parameters
    ----------
    use_hybrid : bool
        If True, use lightweight TF-IDF pre-filtering (TextTiling) to identify
        candidate topic-change points, then only run Sentence-BERT on those
        neighborhoods. This can cut SBERT encoding calls by 50-70% on long
        lectures with stable topic stretches.

    Returns
    -------
    signal : np.ndarray of shape (T,)
        signal[0] = 0; signal[t] = 1 - cosine(embed(window_t), embed(window_{t-1}))
    window_texts : List[str]
        Per-time-step transcript text (used later for segment aggregation).
    """
    window_texts = [text_for_window(transcript, t, half_window) for t in timestamps]
    T = len(timestamps)

    if use_hybrid and T > 20:
        # --- Hybrid mode: TF-IDF pre-filter + selective SBERT ---
        mask = texttiling_prefilter(window_texts)
        encode_indices = np.where(mask)[0]

        pct_encoded = 100 * len(encode_indices) / T
        print(f"    Hybrid topic drift: encoding {len(encode_indices)}/{T} "
              f"steps with SBERT ({pct_encoded:.0f}%), rest interpolated")

        # Encode only selected steps
        selected_texts = [window_texts[i] for i in encode_indices]
        selected_embeds = encoder.encode(selected_texts)  # (N, D)

        # Build full embedding array by interpolating for non-encoded steps
        D = selected_embeds.shape[1]
        embeds = np.zeros((T, D), dtype=np.float32)

        # Map encoded embeddings to their positions
        for j, idx in enumerate(encode_indices):
            embeds[idx] = selected_embeds[j]

        # Interpolate: for non-encoded steps, use nearest encoded neighbor
        encoded_set = set(encode_indices)
        last_encoded = 0
        for t in range(T):
            if t in encoded_set:
                last_encoded = t
            else:
                embeds[t] = embeds[last_encoded]
    else:
        # --- Original mode: encode everything ---
        embeds = encoder.encode(window_texts)

    signal = np.zeros(T, dtype=np.float32)
    if len(embeds) > 1:
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = embeds / norms
        cos_sim = np.sum(unit[1:] * unit[:-1], axis=1)
        signal[1:] = np.clip(1.0 - cos_sim, 0.0, 2.0)
    return signal, window_texts