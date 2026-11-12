"""
Concrete LLM providers. All use plain HTTP (no vendor SDKs), so they are easy
to read and have no extra dependencies.

  - Groq   : OpenAI-compatible chat API, free tier, very fast inference
  - Gemini : Google Generative Language API, free tier
  - Ollama : local models on your own machine (no key, no internet)
  - Mock   : scripted answers, used by the tests (never calls a network)

Model names change over time; override the defaults with LLM_MODEL.
"""

import json
import os
import time
from typing import Callable, List, Optional

from src.llm.base import LLMClient, LLMError, MissingAPIKeyError, RateLimitError, post_json


def _timeout() -> float:
    return float(os.environ.get("LLM_TIMEOUT", "30"))


class GroqClient(LLMClient):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"
    MODELS_URL = "https://api.groq.com/openai/v1/models"
    # Preference order when the configured model has been retired by Groq
    # (Groq retires models regularly; llama-3.1-8b-instant disappeared in 2026).
    PREFERRED = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b",
                 "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("GROQ_API_KEY is not set. Create a free key at console.groq.com and export it.")
        self.model = model or os.environ.get("LLM_MODEL", self.PREFERRED[0])
        self._cooldown = {}  # model -> time before which it is skipped after a 429

    def _pick_available_model(self) -> Optional[str]:
        import requests
        try:
            r = requests.get(self.MODELS_URL, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=_timeout())
            ids = {m["id"] for m in r.json().get("data", []) if m.get("active", True)} if r.ok else set()
        except Exception:
            return None
        return next((m for m in self.PREFERRED if m in ids and m != self.model), None)

    def _fallback_models(self) -> List[str]:
        # Groq quotas (tokens per minute AND per day) are per model, so another model
        # usually still has budget when the main one is exhausted.
        env = os.environ.get("LLM_FALLBACK_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-120b")
        return [m.strip() for m in env.split(",") if m.strip() and m.strip() != self.model]

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        now = time.time()
        candidates = [m for m in [self.model] + self._fallback_models() if self._cooldown.get(m, 0) <= now]
        last_err = None
        for model in candidates or [self.model]:
            try:
                out = self._complete_with_retired_check(system, user, json_mode, model)
                self.last_stats["model"] = model
                return out
            except RateLimitError as e:
                last_err = e
                # Skip this model for a while: until tomorrow's reset if the daily quota is gone,
                # otherwise for about a minute (tokens-per-minute window).
                daily = self.last_stats.get("daily_limit")
                self._cooldown[model] = now + (15 * 60 if daily else max(getattr(e, "retry_after", None) or 0, 60))
                print(f"[GroqClient] {model} is rate-limited ({'daily quota' if daily else 'per-minute'}); "
                      f"trying the next model.")
            except LLMError as e:
                if model == self.model or ("model_not_found" not in str(e) and "decommissioned" not in str(e)):
                    raise
                last_err = e  # a fallback model was retired: try the next one
        raise last_err

    def _complete_with_retired_check(self, system: str, user: str, json_mode: bool, model: str) -> str:
        if model != self.model:
            return self._complete(system, user, json_mode, model)
        try:
            return self._complete(system, user, json_mode)
        except LLMError as e:
            if isinstance(e, RateLimitError):
                raise
            if "model_not_found" not in str(e) and "decommissioned" not in str(e):
                raise
            replacement = self._pick_available_model()
            if not replacement:
                raise LLMError(f"Groq model '{self.model}' is not available and no known replacement was found. "
                               f"Set LLM_MODEL in .env to a model listed at console.groq.com/docs/models.") from e
            print(f"[GroqClient] Model '{self.model}' is unavailable; switching to '{replacement}'. "
                  f"Set LLM_MODEL={replacement} in .env to make this permanent.")
            self.model = replacement
            return self._complete(system, user, json_mode)

    def _complete(self, system: str, user: str, json_mode: bool = True, model: Optional[str] = None) -> str:
        model = model or self.model
        payload = {"model": model, "temperature": 0.0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss"):
            # Hidden reasoning tokens count against the quota; grounded answers need little.
            payload["reasoning_effort"] = os.environ.get("LLM_REASONING_EFFORT", "low")
        self.last_stats = {}
        data = post_json(self.URL, payload, {"Authorization": f"Bearer {self.api_key}"}, _timeout(),
                         provider="Groq", stats=self.last_stats)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Groq response shape: {str(data)[:200]}") from e


class GeminiClient(LLMClient):
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("GEMINI_API_KEY is not set. Create a free key at aistudio.google.com and export it.")
        self.model = model or os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"temperature": 0.0}}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        self.last_stats = {}
        data = post_json(self.URL.format(model=self.model), payload, {"x-goog-api-key": self.api_key},
                         _timeout(), provider="Gemini", stats=self.last_stats)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini response (possibly blocked): {str(data)[:200]}") from e


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "llama3.2:3b")

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        payload = {"model": self.model, "stream": False, "options": {"temperature": 0.0},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["format"] = "json"
        self.last_stats = {}
        data = post_json(f"{self.host}/api/chat", payload, timeout=max(_timeout(), 120), provider="Ollama",
                         stats=self.last_stats)
        return data.get("message", {}).get("content", "")


class MockLLMClient(LLMClient):
    """Returns scripted outputs in order (or from a function of the prompt).
    Records every call so tests can inspect the prompts."""
    name = "mock"
    model = "mock"

    def __init__(self, outputs: Optional[List[str]] = None, fn: Optional[Callable[[str, str], str]] = None):
        self.outputs = list(outputs or [])
        self.fn = fn
        self.calls = []

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        self.calls.append({"system": system, "user": user})
        if self.fn:
            return self.fn(system, user)
        if self.outputs:
            return self.outputs.pop(0)
        return json.dumps({"answer": "", "citations": [], "covered": False})
