"""
Citation validation.

An LLM can cite a label that was never in its context (a hallucinated
source). Every citation is therefore checked against the labels the pipeline
actually supplied: unknown labels are dropped, both from the citation list and
from the answer text, and valid ones are turned into clickable references
like "[Lecture 3, 12:45]" that open the video at that moment.
"""

import re
from typing import Dict, List, Tuple

from src.rag.schema import Citation, LLMAnswer
from src.utils.io_utils import format_timestamp

LABEL_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def video_link(hit: Dict) -> str:
    start = int(hit["start_time"])
    url = hit.get("url") or f"/videos/{hit['video_id']}.mp4"
    if "youtu" in url:
        return url + ("&" if "?" in url else "?") + f"t={start}s"
    return f"{url}#t={start}"


def make_citation(label: str, hit: Dict) -> Citation:
    ts = format_timestamp(hit["start_time"])
    title = hit.get("title", hit["video_id"])
    return Citation(label=label, segment_id=hit["segment_id"], video_id=hit["video_id"], title=title,
                    start_time=float(hit["start_time"]), end_time=float(hit["end_time"]), timestamp=ts,
                    display=f"[{title}, {ts}]", url=video_link(hit), snippet=(hit.get("transcript_text") or "")[:240])


def validate_citations(parsed: LLMAnswer, labels: Dict[str, Dict]) -> Tuple[str, List[Citation], List[str]]:
    """Returns (answer text with readable citations, valid citations, dropped labels)."""
    cited = list(dict.fromkeys(parsed.citations + [x.strip() for m in LABEL_RE.findall(parsed.answer)
                                                    for x in m.split(",")]))
    valid = [c for c in cited if c in labels]
    dropped = [c for c in cited if c not in labels]

    def repl(m):
        keep = [x.strip() for x in m.group(1).split(",") if x.strip() in labels]
        return " ".join(f"[{labels[x].get('title', labels[x]['video_id'])}, {format_timestamp(labels[x]['start_time'])}]"
                        for x in keep)

    # A sentence whose ONLY citations were invalid is unsupported: remove it
    # rather than present it without a source.
    kept_sentences = []
    for sent in re.split(r"(?<=[.!?])\s+", parsed.answer.strip()):
        cited_here = [x.strip() for m in LABEL_RE.findall(sent) for x in m.split(",")]
        if cited_here and not any(x in labels for x in cited_here):
            continue
        kept_sentences.append(sent)
    text = re.sub(r"\s+([.,;:])", r"\1", LABEL_RE.sub(repl, " ".join(kept_sentences))).strip()
    return text, [make_citation(c, labels[c]) for c in valid], dropped
