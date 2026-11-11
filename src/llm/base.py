"""
The LLMClient interface and the HTTP plumbing shared by every provider.

Every provider exposes one method:

    complete(system: str, user: str, json_mode: bool = True) -> str

so the RAG pipeline never knows (or cares) which model is behind it. HTTP
calls go through `post_json`, which adds a timeout, retries with exponential
backoff on rate limits / server errors / timeouts, and turns failures into a
few clear exception types the API and UI can show to a user.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests


class LLMError(RuntimeError):
    """Any failure talking to the LLM provider."""


class MissingAPIKeyError(LLMError):
    """The provider needs an API key that is not set in the environment."""


class RateLimitError(LLMError):
    """The provider kept returning HTTP 429 after all retries."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the timeout after all retries."""


class LLMClient(ABC):
    name: str = "base"
    model: str = ""
    last_stats: Dict = {}  # filled by HTTP providers: request_ms, attempts, rate_limited

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        """Return the model's raw text answer."""


def post_json(url: str, payload: Dict, headers: Optional[Dict] = None, timeout: float = 30.0,
              retries: Optional[int] = None, backoff: float = 1.5, provider: str = "LLM",
              stats: Optional[Dict] = None) -> Dict:
    """POST JSON with timeout + retries (429 / 5xx / timeout), exponential
    backoff with jitter, honouring Retry-After when the server sends it.

    If `stats` is given it is filled with: request_ms (duration of the
    successful attempt only), attempts, and rate_limited (number of 429s), so
    callers can separate model latency from time spent waiting on limits."""
    import os
    retries = int(os.environ.get("LLM_RETRIES", "3")) if retries is None else retries
    stats = stats if stats is not None else {}
    stats.update({"attempts": 0, "rate_limited": 0, "request_ms": None})
    last_err = None
    for attempt in range(retries + 1):
        stats["attempts"] = attempt + 1
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        except requests.Timeout as e:
            last_err = LLMTimeoutError(f"{provider} did not respond within {timeout:.0f}s")
        except requests.RequestException as e:
            last_err = LLMError(f"Could not reach {provider}: {e}")
        else:
            if r.status_code == 200:
                stats["request_ms"] = (time.perf_counter() - t0) * 1000
                return r.json()
            if r.status_code in (401, 403):
                raise LLMError(f"{provider} rejected the API key (HTTP {r.status_code}). Check the key in your environment.")
            if r.status_code == 429:
                stats["rate_limited"] += 1
                last_err = RateLimitError(f"{provider} rate limit hit (HTTP 429). Wait a minute or use another provider.")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < retries:
                    time.sleep(min(float(retry_after), 30.0))
                    continue
            elif r.status_code >= 500:
                last_err = LLMError(f"{provider} server error (HTTP {r.status_code}).")
            else:
                raise LLMError(f"{provider} request failed (HTTP {r.status_code}): {r.text[:300]}")
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, 0.5))
    raise last_err
