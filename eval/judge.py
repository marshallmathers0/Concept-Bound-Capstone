"""
LLM-as-judge for faithfulness.

The judge sees ONLY the segments the answer cited and the answer itself, and
labels each claim supported / unsupported with a fixed rubric (prompts.py).
Faithfulness = supported claims / total claims, averaged over answers.

Limitations (also in LEARNING.md): the judge is itself an LLM, so it can be
lenient towards fluent text, may share biases with the generator when it is
the same model family, splits claims inconsistently, and is not a substitute
for spot-checking answers by hand.
"""

from typing import Dict, List, Optional

from src.rag.context import build_context
from src.rag.prompts import JUDGE_SYSTEM, build_judge_prompt


def judge_faithfulness(llm, cited_hits: List[Dict], answer: str) -> Optional[float]:
    if not cited_hits or not answer.strip():
        return None
    context, _ = build_context(cited_hits, max_tokens=2500)
    raw = llm.complete(JUDGE_SYSTEM, build_judge_prompt(context, answer), json_mode=True)
    import json, re
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    claims = data.get("claims") or []
    if claims:
        return sum(1 for c in claims if c.get("supported")) / len(claims)
    f = data.get("faithfulness")
    return float(f) if isinstance(f, (int, float)) else None
