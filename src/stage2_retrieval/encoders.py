"""
Frozen encoders used by Stage 2:
  - TextEncoder: wraps Sentence-BERT for OCR text, transcript text and the
    student's free-text query.
  - The CLIP image encoder is reused from `src.feature_extraction.clip_extractor`
    for representative-keyframe embeddings.

These are NOT trained; only the SegmentFusionEncoder and QueryProjector
(in `fusion_model.py`) are trainable in Stage 2.
"""

from typing import List
import numpy as np

from src.feature_extraction.asr_extractor import SentenceEncoder
from src.feature_extraction.clip_extractor import CLIPEncoder
from config import MODELS


class TextEncoder:
    """Thin convenience wrapper so Stage 2 modules share one Sentence-BERT instance."""

    def __init__(self, model_name: str = MODELS.sentence_bert, device: str = None):
        self._sbert = SentenceEncoder(model_name, device=device)

    @property
    def dim(self) -> int:
        return self._sbert.dim

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._sbert.encode(texts)

    def encode_long(self, texts: List[str]) -> np.ndarray:
        """Chunk-and-average encoding for segment-length texts."""
        return self._sbert.encode_long(texts)


class VisualEncoder:
    """Thin convenience wrapper around the frozen CLIP image encoder."""

    def __init__(self, model_name: str = MODELS.clip_model, device: str = None):
        self._clip = CLIPEncoder(model_name, device=device)

    @property
    def dim(self) -> int:
        return self._clip.dim

    def encode_images(self, frames) -> np.ndarray:
        return self._clip.encode_images(frames)
