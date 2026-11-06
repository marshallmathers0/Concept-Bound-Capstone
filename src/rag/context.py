"""
Context builder: turn retrieved segments into the text block the LLM reads.

  - de-duplicates overlapping segments (same lecture, >50% time overlap),
    keeping the higher-ranked one
  - labels each segment [S1], [S2], ... (short labels are copied far more
    reliably by LLMs than long ids; the pipeline maps them back to segment ids)
  - stays inside a token budget, trimming the lowest-ranked segments' text
    first (token count estimated as words * 1.3, good enough for budgeting)
"""

from typing import Dict, List, Tuple

from src.utils.io_utils import format_timestamp


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def _overlap_frac(a: Dict, b: Dict) -> float:
    if a["video_id"] != b["video_id"]:
        return 0.0
    inter = max(0.0, min(a["end_time"], b["end_time"]) - max(a["start_time"], b["start_time"]))
    shorter = max(1e-6, min(a["end_time"] - a["start_time"], b["end_time"] - b["start_time"]))
    return inter / shorter


def dedupe(hits: List[Dict], max_overlap: float = 0.5) -> List[Dict]:
    kept: List[Dict] = []
    for h in hits:  # hits arrive best-first
        if all(_overlap_frac(h, k) <= max_overlap for k in kept):
            kept.append(h)
    return kept


def build_context(hits: List[Dict], max_tokens: int = 1800, per_segment_tokens: int = 450) -> Tuple[str, Dict[str, Dict]]:
    """Returns (context_text, {"S1": hit, ...})."""
    blocks, labels, used = [], {}, 0
    for i, h in enumerate(dedupe(hits), start=1):
        label = f"S{i}"
        header = (f"[{label}] {h.get('title', h['video_id'])} — "
                  f"{format_timestamp(h['start_time'])}–{format_timestamp(h['end_time'])}")
        slide = (h.get("ocr_text") or "").strip()
        speech = (h.get("transcript_text") or "").strip()
        budget = min(per_segment_tokens, max_tokens - used - estimate_tokens(header))
        if budget < 40:
            break
        words = (("Slide text: " + slide + "\n") if slide else "") + "Transcript: " + speech
        words = " ".join(words.split(" ")[: int(budget / 1.3)])
        block = header + "\n" + words
        blocks.append(block)
        labels[label] = h
        used += estimate_tokens(block)
    return "\n\n".join(blocks), labels
