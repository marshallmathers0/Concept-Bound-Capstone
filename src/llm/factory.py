"""Pick the LLM provider from the environment (LLM_PROVIDER=groq|gemini|ollama|none)."""

import os
from typing import Optional

from src.llm.base import LLMClient


def get_llm_client(provider: Optional[str] = None) -> Optional[LLMClient]:
    """Returns None when no provider is configured: the app then runs in
    retrieval-only mode instead of failing."""
    try:
        from dotenv import load_dotenv
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(root, ".env"))
    except ImportError:
        pass
    provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if not provider:
        # Auto-detect from whichever key is present.
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            return None
    if provider in ("none", "off"):
        return None
    if provider == "groq":
        from src.llm.providers import GroqClient
        return GroqClient()
    if provider == "gemini":
        from src.llm.providers import GeminiClient
        return GeminiClient()
    if provider == "ollama":
        from src.llm.providers import OllamaClient
        return OllamaClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use groq, gemini, ollama or none)")
