"""
Search service used by the RAG pipeline, the API and the eval.

It reuses the existing LectureRetriever (FAISS fusion index + BM25 +
Sentence-BERT) unchanged and adds two things:

  1. optional cross-encoder reranking of the top candidates
  2. a confidence score for the refusal gate:
       - without reranker: cosine similarity between the query and the best
         matching segment in the WHOLE collection (Sentence-BERT)
       - with reranker: sigmoid of the reranker's top logit
     If confidence is below the threshold, the pipeline refuses instead of
     letting the LLM guess. Thresholds are calibrated on held-out questions
     by `python -m eval.run` (stored in <index>/thresholds.json).
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import PATHS

DEFAULT_INDEX = os.environ.get("RAG_INDEX_DIR", os.path.join(os.path.dirname(PATHS.index_dir), "rag_index"))
DEFAULT_THRESHOLDS = {"dense": 0.35, "rerank": 0.10}  # used until eval calibrates them
MODES = ("bm25", "dense", "hybrid", "model", "auto")


@dataclass
class SearchResult:
    hits: List[Dict]
    confidence: float
    confidence_source: str
    timings_ms: Dict[str, float] = field(default_factory=dict)


class SearchService:
    def __init__(self, index_dir: str = DEFAULT_INDEX, use_reranker: bool = True, device: str = "cpu"):
        from src.stage2_retrieval.retrieval import LectureRetriever
        self.index_dir = index_dir
        self.retriever = LectureRetriever(index_dir=index_dir, device=device)
        self.by_id = {m.get("segment_id"): m for m in self.retriever.metadata}
        self.reranker = None
        if use_reranker:
            from src.retrieval.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(device=device)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        path = os.path.join(index_dir, "thresholds.json")
        if os.path.exists(path):
            self.thresholds.update(json.load(open(path)))
        if os.environ.get("RAG_MIN_CONFIDENCE"):
            v = float(os.environ["RAG_MIN_CONFIDENCE"])
            self.thresholds = {"dense": v, "rerank": v}

    def warm_up(self):
        """Load the lazily-initialized encoders now instead of on the first request."""
        self.search("warm up", top_k=1)

    def dense_confidence(self, query: str) -> float:
        q = self.retriever.text_encoder.encode([query])[0]
        q = q / (np.linalg.norm(q) + 1e-8)
        return float(np.max(self.retriever.segment_text_embeddings @ q))

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", rerank: Optional[bool] = None,
               candidates: int = 20, rerank_order: str = "transcript_first") -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        rerank = (self.reranker is not None) if rerank is None else rerank
        if rerank and self.reranker is None:
            raise RuntimeError("Reranking requested but the reranker is not loaded (use_reranker=False).")
        t0 = time.perf_counter()
        hits = self.retriever.search(query, top_k=candidates if rerank else top_k, mode=mode)
        for h in hits:  # attach full metadata (times, texts, title) for downstream steps
            h.update({k: v for k, v in self.by_id.get(h["segment_id"], {}).items() if k not in h})
            h["retrieval_score"] = h["score"]
        conf = self.dense_confidence(query)
        timings = {"retrieval_ms": (time.perf_counter() - t0) * 1000}
        source = "dense"
        if rerank:
            t1 = time.perf_counter()
            hits = self.reranker.rerank(query, hits, order=rerank_order)
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
            if hits:
                conf, source = hits[0]["rerank_prob"], "rerank"
        return SearchResult(hits=hits[:top_k], confidence=conf, confidence_source=source, timings_ms=timings)

    def threshold(self, source: str) -> float:
        return self.thresholds.get(source, DEFAULT_THRESHOLDS[source])
