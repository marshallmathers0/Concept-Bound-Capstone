"""
The RAG pipeline:

  question (+history) -> rewrite follow-ups -> hybrid retrieval (existing)
    -> cross-encoder rerank -> confidence gate (refuse if low)
    -> context builder -> LLM (JSON) -> Pydantic parse -> citation validation
    -> answer with [Lecture, mm:ss] links + per-stage latency

Without an LLM configured it runs in retrieval-only mode and returns the
top segments, so the demo still works with no API key.
"""

import time
from collections import OrderedDict
from typing import List, Optional

from src.llm.base import LLMClient, LLMError
from src.rag.citations import make_citation, validate_citations
from src.rag.context import build_context
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.rag.rewrite import rewrite_query
from src.rag.schema import AskResponse, RetrievedSegment, parse_llm_output
from src.retrieval.search import SearchService

REFUSAL = "This isn't covered in the indexed lectures."


class RAGPipeline:
    def __init__(self, search: SearchService, llm: Optional[LLMClient] = None, mode: str = "hybrid",
                 candidates: int = 20, context_k: int = 5, cache_size: int = 256):
        self.search, self.llm, self.mode = search, llm, mode
        self.candidates, self.context_k = candidates, context_k
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, str]" = OrderedDict()  # user prompt -> raw LLM output (LRU)

    def ask(self, question: str, history: Optional[List] = None, use_reranker: Optional[bool] = None,
            top_k: Optional[int] = None) -> AskResponse:
        t_all = time.perf_counter()
        history = history or []
        top_k = top_k or self.context_k
        lat = {}

        t = time.perf_counter()
        query = rewrite_query(question, history, self.llm)
        lat["rewrite_ms"] = (time.perf_counter() - t) * 1000

        res = self.search.search(query, top_k=top_k, mode=self.mode, rerank=use_reranker, candidates=self.candidates)
        lat.update(res.timings_ms)
        threshold = self.search.threshold(res.confidence_source)
        retrieved = [RetrievedSegment(segment_id=h["segment_id"], title=h.get("title", h["video_id"]),
                                      timestamp=h.get("start_str", ""), retrieval_score=float(h["retrieval_score"]),
                                      rerank_score=h.get("rerank_score"), snippet=(h.get("transcript_text") or "")[:200])
                     for h in res.hits]

        def respond(answer, citations, mode, refused=False, dropped=None, error=None):
            lat["total_ms"] = (time.perf_counter() - t_all) * 1000
            return AskResponse(question=question, rewritten_query=query, answer=answer, citations=citations,
                               refused=refused, confidence=res.confidence, confidence_threshold=threshold,
                               mode=mode, latency_ms={k: round(v, 1) for k, v in lat.items()},
                               retrieved=retrieved, dropped_citations=dropped or [], error=error)

        # Refusal gate: reuse retrieval confidence instead of letting the LLM guess.
        if not res.hits or res.confidence < threshold:
            return respond(REFUSAL, [], "refused", refused=True)

        context, labels = build_context(res.hits)
        if self.llm is None:
            cites = [make_citation(lbl, h) for lbl, h in labels.items()]
            return respond("No LLM is configured (set LLM_PROVIDER and an API key); showing the most relevant "
                           "lecture moments instead.", cites, "retrieval_only")

        t = time.perf_counter()
        user_prompt = build_user_prompt(question if query == question else f"{question} (context: {query})", context)
        cached = self._cache.get(user_prompt)
        try:
            if cached is not None:
                # Same question over the same retrieved context: temperature 0, so reuse the answer
                # instead of spending rate-limited tokens again (demo example buttons hit this a lot).
                raw = cached
                self._cache.move_to_end(user_prompt)
                lat["llm_cached"] = 1.0
            else:
                raw = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        except LLMError as e:
            lat["llm_ms"] = (time.perf_counter() - t) * 1000
            cites = [make_citation(lbl, h) for lbl, h in labels.items()]
            return respond(f"The language model is unavailable ({e}). Showing the most relevant lecture moments.",
                           cites, "llm_error", error=str(e))
        lat["llm_ms"] = (time.perf_counter() - t) * 1000
        st = {} if cached is not None else (getattr(self.llm, "last_stats", None) or {})
        if st.get("request_ms") is not None:
            lat["llm_request_ms"] = st["request_ms"]          # model time only
            lat["llm_rate_limited"] = float(st.get("rate_limited", 0))  # 429s hit on the way

        parsed, err = parse_llm_output(raw)
        if parsed is not None and cached is None:
            self._cache[user_prompt] = raw
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        if parsed is None:
            cites = [make_citation(lbl, h) for lbl, h in labels.items()]
            return respond("The model's answer could not be parsed, so here are the most relevant lecture "
                           "moments instead.", cites, "llm_error", error=f"parse failure: {err}")
        if not parsed.covered:
            return respond(REFUSAL, [], "refused", refused=True)
        text, cites, dropped = validate_citations(parsed, labels)
        if not cites:
            # An answer with no valid citation is not grounded: do not show it as one.
            return respond(REFUSAL, [], "refused", refused=True, dropped=dropped)
        return respond(text, cites, "rag", dropped=dropped)
