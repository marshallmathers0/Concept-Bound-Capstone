"""
Build the transcript / OCR text of an arbitrary time span of a video.

Every segmenter (Stage 1, fixed windows, gold boundaries) and every retriever
(BM25, dense, fusion model) goes through these helpers, so all systems see
exactly the same text for the same span.

Why not just join the per-step `transcript_texts`? Those are *rolling 30-second
windows* sampled every 3 seconds, so joining them repeats every word ~10 times.
That inflates BM25 term counts and, worse, means a Sentence-BERT encoder (256
token limit) only ever sees the first few seconds of a segment.
"""

from typing import Dict, List

import numpy as np


def has_raw_asr(feats: Dict) -> bool:
    return "asr_segments" in feats and len(feats["asr_segments"]) > 0


def span_transcript(feats: Dict, start_time: float, end_time: float) -> str:
    """Transcript spoken in [start_time, end_time).

    Uses the raw Whisper segments (assigned by their midpoint) when the
    feature file has them. Older feature files only have the rolling windows;
    for those, take every k-th window so windows do not overlap, which removes
    most (not all) duplication. Re-run `scripts/refresh_features.py --asr` to
    get raw segments.
    """
    if has_raw_asr(feats):
        parts = []
        for seg in clean_asr(feats):
            mid = 0.5 * (seg.start + seg.end)
            if start_time <= mid < end_time:
                parts.append(seg.text)
        return " ".join(parts).strip()

    timestamps = feats["timestamps"]
    windows = feats["transcript_texts"]
    if len(timestamps) < 2:
        return " ".join(str(w) for w in windows).strip()
    step = float(timestamps[1] - timestamps[0])
    stride = max(1, int(round(30.0 / step)))  # rolling window is 30s wide
    idx = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
    picked = idx[::stride] if len(idx) else idx
    return _dedupe_sentences(" ".join(str(windows[i]) for i in picked))


def clean_asr(feats: Dict):
    """Raw Whisper segments with looped hallucinations removed. Cached on the
    feats dict; also applied to files written before the filter existed."""
    if "_clean_asr" not in feats:
        from src.feature_extraction.asr_extractor import asr_segments_from_array, drop_repeated_segments
        feats["_clean_asr"] = drop_repeated_segments(asr_segments_from_array(feats["asr_segments"]))
    return feats["_clean_asr"]


def span_ocr(feats: Dict, start_time: float, end_time: float) -> str:
    """Distinct slide text visible in [start_time, end_time).

    Per-step OCR text is carried forward while a slide is unchanged, so only
    keep a frame's text when it differs from the previous frame's.
    """
    timestamps = feats["timestamps"]
    texts = feats["ocr_texts"]
    idx = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
    out: List[str] = []
    prev = None
    for i in idx:
        t = str(texts[i]).strip()
        if t and t != prev:
            out.append(t)
        prev = t
    return " ".join(out)


def _dedupe_sentences(text: str) -> str:
    seen, out = set(), []
    for sent in text.replace("?", ".").replace("!", ".").split("."):
        key = sent.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(sent.strip())
    return ". ".join(out)
