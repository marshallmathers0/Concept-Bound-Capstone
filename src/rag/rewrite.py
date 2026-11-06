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

FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|he|she|more|again|simpler|simply|"
                       r"another|example|elaborate|why|how so|same|above|previous)\b", re.I)


def is_follow_up(question: str) -> bool:
    return len(question.split()) <= 6 or bool(FOLLOW_UP.search(question))


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
