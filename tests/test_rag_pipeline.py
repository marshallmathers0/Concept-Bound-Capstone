"""Pipeline behaviour with a mocked LLM: grounding, refusal, citation validation, parsing."""

import json

from conftest import FakeSearch, make_hit
from src.llm.base import LLMError, RateLimitError
from src.llm.providers import MockLLMClient
from src.rag.citations import validate_citations
from src.rag.pipeline import REFUSAL, RAGPipeline
from src.rag.schema import LLMAnswer, parse_llm_output


def llm_json(answer, citations, covered=True):
    return MockLLMClient([json.dumps({"answer": answer, "citations": citations, "covered": covered})])


# ---- citation validation -------------------------------------------------------------

def test_citation_validation_drops_fake_ids():
    labels = {"S1": make_hit(1), "S2": make_hit(2)}
    parsed = LLMAnswer(answer="Real fact [S1]. Invented fact [S9]. Mixed [S2, S7].", citations=["S1", "S9", "S2"])
    text, cites, dropped = validate_citations(parsed, labels)
    assert [c.label for c in cites] == ["S1", "S2"]
    assert set(dropped) == {"S9", "S7"}
    assert "Invented fact" not in text  # sentence supported only by a fake id is removed
    assert "[Lecture T, 02:00]" in text and "[Lecture T, 04:00]" in text
    assert "S9" not in text and "S7" not in text


def test_citation_links_open_video_at_timestamp():
    labels = {"S1": make_hit(1, start=765.0)}
    _, cites, _ = validate_citations(LLMAnswer(answer="x [S1].", citations=["S1"]), labels)
    assert cites[0].url.endswith("#t=765") and cites[0].display == "[Lecture T, 12:45]"
    yt = dict(make_hit(1, start=765.0), url="https://www.youtube.com/watch?v=abcdefghijk")
    _, cites, _ = validate_citations(LLMAnswer(answer="x [S1].", citations=["S1"]), {"S1": yt})
    assert cites[0].url.endswith("&t=765s")


def test_pipeline_end_to_end_with_mock_llm(fake_search):
    llm = llm_json("Topic one is explained here [S1]. Also [S42].", ["S1", "S42"])
    r = RAGPipeline(fake_search, llm).ask("what is topic one?")
    assert r.mode == "rag" and not r.refused
    assert [c.segment_id for c in r.citations] == ["lec_test#000"]
    assert r.dropped_citations == ["S42"]
    assert {"retrieval_ms", "rerank_ms", "llm_ms", "total_ms"} <= set(r.latency_ms)
    assert "[S1]" in llm.calls[0]["user"]  # the context carried the label the LLM cited


# ---- refusal ------------------------------------------------------------------------------

def test_low_confidence_refuses_without_calling_llm():
    llm = MockLLMClient()
    r = RAGPipeline(FakeSearch(confidence=0.1, threshold=0.5), llm).ask("who won the 2011 world cup?")
    assert r.refused and r.answer == REFUSAL and r.citations == []
    assert llm.calls == []  # the LLM never got a chance to guess


def test_llm_saying_not_covered_refuses(fake_search):
    r = RAGPipeline(fake_search, llm_json("Not in the lectures.", [], covered=False)).ask("q")
    assert r.refused and r.mode == "refused"


def test_answer_with_only_fake_citations_is_refused(fake_search):
    r = RAGPipeline(fake_search, llm_json("Made up [S9].", ["S9"])).ask("q")
    assert r.refused and r.dropped_citations == ["S9"]


def test_retrieval_only_mode_without_llm(fake_search):
    r = RAGPipeline(fake_search, None).ask("q")
    assert r.mode == "retrieval_only" and len(r.citations) == 5 and not r.refused


def test_llm_error_degrades_gracefully(fake_search):
    class Failing(MockLLMClient):
        def complete(self, *a, **k):
            raise RateLimitError("rate limited")
    r = RAGPipeline(fake_search, Failing()).ask("q")
    assert r.mode == "llm_error" and "rate limited" in r.error and r.citations


# ---- structured output parsing ------------------------------------------------------------------

def test_parse_plain_json():
    p, err = parse_llm_output('{"answer": "a [S1]", "citations": ["S1"], "covered": true}')
    assert err is None and p.citations == ["S1"]


def test_parse_fenced_and_wrapped_json():
    p, err = parse_llm_output('Sure! Here it is:\n```json\n{"answer": "a", "citations": "S2"}\n```\nThanks')
    assert err is None and p.citations == ["S2"] and p.covered is True


def test_parse_coerces_bracketed_labels():
    p, _ = parse_llm_output('{"answer": "a", "citations": ["[S1]", 3]}')
    assert p.citations == ["S1", "3"]


def test_parse_malformed_returns_error_not_exception():
    for bad in ["", "no json here", '{"answer": ', '{"answer": ["not", "a", "string"]}']:
        p, err = parse_llm_output(bad)
        assert p is None and err


def test_pipeline_handles_malformed_llm_output(fake_search):
    r = RAGPipeline(fake_search, MockLLMClient(["I think the answer is 42"])).ask("q")
    assert r.mode == "llm_error" and "parse failure" in r.error and r.citations
