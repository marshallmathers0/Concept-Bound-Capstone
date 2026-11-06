import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.retrieval.search import SearchResult  # noqa: E402


def make_hit(i, video="lec_test", start=None, text=None):
    start = start if start is not None else i * 120.0
    return {"segment_id": f"{video}#{i:03d}", "video_id": video, "title": "Lecture T",
            "url": f"/videos/{video}.mp4", "start_time": start, "end_time": start + 120.0,
            "start_str": f"{int(start // 60):02d}:{int(start % 60):02d}",
            "transcript_text": text or f"segment {i} explains topic {i} in detail",
            "ocr_text": f"slide {i}", "score": 1.0 - i * 0.1, "retrieval_score": 1.0 - i * 0.1}


class FakeSearch:
    """Stands in for SearchService: no models, deterministic hits and confidence."""

    def __init__(self, hits=None, confidence=0.9, source="rerank", threshold=0.5):
        self.hits = hits if hits is not None else [make_hit(i) for i in range(5)]
        self.confidence, self.source, self._threshold = confidence, source, threshold
        self.by_id = {h["segment_id"]: h for h in self.hits}
        self.reranker = object()
        self.queries = []

    def search(self, query, top_k=5, mode="hybrid", rerank=None, candidates=20):
        self.queries.append(query)
        return SearchResult(hits=[dict(h) for h in self.hits[:top_k]], confidence=self.confidence,
                            confidence_source=self.source, timings_ms={"retrieval_ms": 1.0, "rerank_ms": 1.0})

    def threshold(self, source):
        return self._threshold


@pytest.fixture
def fake_search():
    return FakeSearch()
