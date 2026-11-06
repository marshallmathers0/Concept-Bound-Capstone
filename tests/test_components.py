"""Context builder, follow-up rewriting, LLM factory and HTTP retry logic (no network)."""

import pytest

from conftest import make_hit
from src.llm import base as llm_base
from src.llm.base import LLMError, MissingAPIKeyError, RateLimitError, post_json
from src.llm.factory import get_llm_client
from src.rag.context import build_context, dedupe
from src.rag.rewrite import is_follow_up, rewrite_query


def test_dedupe_drops_overlapping_segments():
    a, b, c = make_hit(0, start=0), make_hit(1, start=30), make_hit(2, start=600)
    assert [h["segment_id"] for h in dedupe([a, b, c])] == [a["segment_id"], c["segment_id"]]


def test_context_respects_token_budget():
    long = " ".join(["word"] * 2000)
    hits = [make_hit(i, start=i * 200, text=long) for i in range(5)]
    ctx, labels = build_context(hits, max_tokens=600, per_segment_tokens=300)
    assert len(ctx.split()) * 1.3 <= 600 + 50
    assert list(labels)[0] == "S1" and len(labels) < 5


def test_follow_up_rewrite_uses_history():
    hist = [{"role": "user", "content": "what is gradient descent"}, {"role": "assistant", "content": "..."}]
    assert is_follow_up("explain it more simply")
    out = rewrite_query("explain it more simply", hist)
    assert "gradient descent" in out and "simply" in out
    standalone = "what is the difference between an encoder model and a decoder model"
    assert rewrite_query(standalone, hist) == standalone


def test_factory_without_keys_returns_none(monkeypatch):
    for k in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    assert get_llm_client() is None


def test_explicit_provider_without_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_llm_client("groq")


class FakeResp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = code, body or {}, headers or {}, str(body)

    def json(self):
        return self._body


def test_post_json_retries_rate_limit_then_succeeds(monkeypatch):
    calls = iter([FakeResp(429), FakeResp(503), FakeResp(200, {"ok": True})])
    monkeypatch.setattr(llm_base.requests, "post", lambda *a, **k: next(calls))
    monkeypatch.setattr(llm_base.time, "sleep", lambda s: None)
    assert post_json("http://x", {}, retries=3, provider="Test") == {"ok": True}


def test_post_json_gives_clear_errors(monkeypatch):
    monkeypatch.setattr(llm_base.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_base.requests, "post", lambda *a, **k: FakeResp(429))
    with pytest.raises(RateLimitError):
        post_json("http://x", {}, retries=2, provider="Test")
    monkeypatch.setattr(llm_base.requests, "post", lambda *a, **k: FakeResp(401))
    with pytest.raises(LLMError, match="API key"):
        post_json("http://x", {}, retries=2, provider="Test")
