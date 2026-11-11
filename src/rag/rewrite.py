"""
Follow-up question rewriting.

"Explain it more simply" means nothing to a retriever on its own. If the new
question looks like a follow-up (it is short, or refers back with words like
"it", "this", "that", "more", "again"), we prepend the previous user
questions (last 2-3 turns) so retrieval sees the topic. Deterministic and
explainable; when an LLM is configured and RAG_LLM_REWRITE=1, the LLM writes
a standalone question instead.
"""

import os
import re
from typing import List, Optional

# Words/phrases that point back to the previous turn. Deliberately NOT plain
# "why"/"how"/"example": "Why is softmax used?" is a standalone question.
FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|the same|the above|the previous)\b"
    r"|\b(more simply|in simpler|simpler terms|explain more|tell me more|more detail|elaborate|"
    r"again|another example|give an example|an example of it|how so|why is that|what about)\b", re.I)


def is_follow_up(question: str) -> bool:
    """Short questions ("and RNNs?") or ones that refer back to the last turn."""
    return len(question.split()) <= 4 or bool(FOLLOW_UP.search(question))


def rewrite_query(question: str, history: List, llm=None, max_turns: int = 3) -> str:
    users = [t.content if hasattr(t, "content") else t["content"]
             for t in history if (t.role if hasattr(t, "role") else t["role"]) == "user"][-max_turns:]
    if not users or not is_follow_up(question):
        return question
    if llm is not None and os.environ.get("RAG_LLM_REWRITE") == "1":
        try:
            out = llm.complete("Rewrite the student's last message as one standalone search question about the "
                               "lecture topic. Return only the question.",
                               "Earlier questions:\n" + "\n".join(users) + f"\nLast message: {question}",
                               json_mode=False).strip()
            if out:
                return out
        except Exception:
            pass  # fall back to the heuristic
    return " ".join(users[-2:] + [question])
