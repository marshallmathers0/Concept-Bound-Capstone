"""
Pydantic models: the LLM's structured output, and the /ask API contract.

The LLM is asked for JSON {"answer": str, "citations": [str], "covered": bool}.
LLMs sometimes wrap JSON in ```fences```, add prose around it, or return the
wrong types, so `parse_llm_output` extracts the first JSON object it can find
and validates it; on failure it returns None plus the reason instead of
raising, and the pipeline degrades gracefully.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


class LLMAnswer(BaseModel):
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    covered: bool = True

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, int)):
            v = [v]
        return [str(x).strip().strip("[]") for x in v if str(x).strip()]


def parse_llm_output(text: str) -> Tuple[Optional[LLMAnswer], Optional[str]]:
    if not text or not text.strip():
        return None, "empty output"
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", cleaned)]:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            try:
                return LLMAnswer.model_validate(obj), None
            except ValidationError as e:
                return None, f"schema error: {e.errors()[0]['msg']}"
    return None, "no JSON object found"


# ---- API models -------------------------------------------------------------------

class Turn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: List[Turn] = Field(default_factory=list)
    use_reranker: Optional[bool] = None
    top_k: int = Field(5, ge=1, le=10)


class Citation(BaseModel):
    label: str               # what the LLM cited, e.g. "S2"
    segment_id: str
    video_id: str
    title: str
    start_time: float
    end_time: float
    timestamp: str           # "mm:ss" of the segment start
    display: str             # "[Lecture 3, 12:45]"
    url: str                 # opens the video at the timestamp
    snippet: str


class RetrievedSegment(BaseModel):
    segment_id: str
    title: str
    timestamp: str
    retrieval_score: float
    rerank_score: Optional[float] = None
    snippet: str


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    citations: List[Citation]
    refused: bool
    confidence: float
    confidence_threshold: float
    mode: str                # "rag", "retrieval_only", "refused", "llm_error"
    latency_ms: Dict[str, float]
    retrieved: List[RetrievedSegment] = Field(default_factory=list)
    dropped_citations: List[str] = Field(default_factory=list)
    error: Optional[str] = None
