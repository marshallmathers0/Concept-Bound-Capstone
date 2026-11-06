"""Retrieval on the real precomputed index (skipped if it has not been built)."""

import os

import pytest

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rag_index")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(INDEX, "segments.index")),
                                reason="run python -m src.rag.build_index first")


@pytest.fixture(scope="module")
def search():
    from src.retrieval.search import SearchService
    return SearchService(index_dir=INDEX, use_reranker=False)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_known_queries_return_results(search, mode):
    res = search.search("recurrent neural network hidden state", top_k=5, mode=mode, rerank=False)
    assert len(res.hits) == 5
    assert all({"segment_id", "video_id", "start_time", "transcript_text"} <= set(h) for h in res.hits)


def test_known_topic_is_found_in_the_right_lecture(search):
    res = search.search("why is softmax used to get probabilities at the output", top_k=5, mode="hybrid", rerank=False)
    assert any(h["video_id"] == "college_lec01" for h in res.hits)


def test_off_topic_query_has_lower_confidence(search):
    on = search.search("what is the vanishing gradient problem in RNNs", rerank=False).confidence
    off = search.search("how do I configure a VLAN on a Cisco switch", rerank=False).confidence
    assert on > off
