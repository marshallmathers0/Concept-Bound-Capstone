"""
Cross-encoder reranker.

The first-stage retrievers are bi-encoders (query and segment embedded
separately) or BM25 (word overlap): fast over the whole collection, but they
never look at the query and the segment together. A cross-encoder reads the
pair jointly through one transformer and outputs a relevance logit, which is
much more accurate but too slow for the whole corpus, so it only re-scores the
top-N candidates (N = 20 here).

Input order matters: the model reads at most 512 tokens. Putting the noisy
OCR text first (the original version) used most of that window on OCR
garbage and the first sentences of the transcript; transcript-first lifted
MRR from 0.556 to 0.666 on the eval set (see reports/eval_results.md).
"""

import math
from typing import Dict, List

from config import MODELS


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODELS.cross_encoder, device: str = "cpu", max_chars: int = 1500):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.max_chars = max_chars

    def rerank(self, query: str, hits: List[Dict], order: str = "transcript_first") -> List[Dict]:
        """Adds `rerank_score` (logit) and `rerank_prob` (sigmoid) to each hit
        and returns them sorted by rerank score."""
        if not hits:
            return []
        def text(h):
            a, b = h.get("transcript_text", ""), h.get("ocr_text", "")
            return ((a + " " + b) if order == "transcript_first" else (b + " " + a))[: self.max_chars]
        pairs = [(query, text(h)) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
            h["rerank_prob"] = 1.0 / (1.0 + math.exp(-float(s)))
        return sorted(hits, key=lambda h: -h["rerank_score"])
