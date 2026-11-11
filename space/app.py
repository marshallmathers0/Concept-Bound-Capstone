"""
ConceptBound - ask questions about lecture videos, get answers with timestamp citations.

Gradio demo for local use and for Hugging Face Spaces (free CPU tier).
It only loads precomputed artifacts (FAISS index, metadata, cached
embeddings); Whisper / OCR never run here.

    python space/app.py                     # local, from the repo root
    (on a Space: app.py at the Space root, built by space/prepare_space.py)

Env: LLM_PROVIDER + GROQ_API_KEY / GEMINI_API_KEY (Space secrets),
     RAG_INDEX_DIR (default: ./rag_index if present, else data/rag_index).
Without an LLM key the demo still works in retrieval-only mode.
"""

import html
import os
import re
import sys

if sys.platform == "win32":
    # Browsers drop keep-alive connections; Windows' Proactor event loop then
    # prints a harmless "ConnectionResetError [WinError 10054]" traceback.
    import asyncio.proactor_events as _pe
    _orig_lost = _pe._ProactorBasePipeTransport._call_connection_lost

    def _quiet_connection_lost(self, exc):
        try:
            _orig_lost(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _quiet_connection_lost

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "src")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)
if not os.environ.get("RAG_INDEX_DIR"):
    local = os.path.join(ROOT, "rag_index")
    os.environ["RAG_INDEX_DIR"] = local if os.path.isdir(local) else os.path.join(ROOT, "data", "rag_index")

import gradio as gr  # noqa: E402

from src.llm.factory import get_llm_client  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.search import SearchService  # noqa: E402

VIDEOS_DIR = os.path.join(ROOT, "data", "raw_videos")
EXAMPLES = [
    "What is the difference between traditional machine learning and deep learning?",
    "Why is softmax used at the output layer?",
    "Why can't a plain RNN remember information from far back in a sentence?",
    "How do you generate a sentence from a bigram model?",
    "What is the difference between decoder-only and encoder-decoder models?",
]

print("Loading index and models ...")
SEARCH = SearchService(index_dir=os.environ["RAG_INDEX_DIR"], use_reranker=True)
SEARCH.warm_up()
try:
    LLM = get_llm_client()
    LLM_STATUS = f"LLM: {LLM.name} ({LLM.model})" if LLM else "LLM: not configured (retrieval-only mode)"
except Exception as e:  # missing key for an explicitly chosen provider
    LLM, LLM_STATUS = None, f"LLM unavailable: {e}"
PIPE = RAGPipeline(SEARCH, LLM)


def youtube_embed(url: str, start: int):
    m = re.search(r"(?:v=|youtu\.be/|embed/)([\w-]{11})", url)
    if not m:
        return None
    return (f'<iframe width="100%" height="220" src="https://www.youtube.com/embed/{m.group(1)}?start={start}" '
            f'frameborder="0" allowfullscreen></iframe>')


def player(c) -> str:
    start = int(c.start_time)
    if "youtu" in c.url:
        return youtube_embed(c.url, start) or f'<a href="{c.url}" target="_blank">Open video at {c.timestamp}</a>'
    local = os.path.join(VIDEOS_DIR, f"{c.video_id}.mp4")
    if os.path.exists(local):
        src = f"/gradio_api/file={local.replace(os.sep, '/')}#t={start}"
        return f'<video width="100%" height="220" controls preload="metadata" src="{src}"></video>'
    return f"<i>Video file not available in this deployment ({html.escape(c.display)}).</i>"


def render_citations(resp) -> str:
    if not resp.citations:
        return ""
    parts = ["<div style='display:flex;flex-direction:column;gap:14px'>"]
    for c in resp.citations:
        parts.append(
            "<div style='border:1px solid #ddd;border-radius:8px;padding:10px'>"
            f"<b>{html.escape(c.display)}</b> <span style='color:#888'>({html.escape(c.segment_id)})</span>"
            f"<p style='font-size:0.9em;margin:6px 0'>&ldquo;{html.escape(c.snippet)}&hellip;&rdquo;</p>"
            f"{player(c)}</div>")
    parts.append("</div>")
    return "".join(parts)


def ask(question, history, use_reranker, show_retrieved):
    question = (question or "").strip()
    if not question:
        return gr.update(), "", "", gr.update(), history
    resp = PIPE.ask(question, history=history, use_reranker=use_reranker)
    answer = resp.answer
    if resp.refused:
        answer = f"**{answer}**\n\n_Retrieval confidence {resp.confidence:.2f} was below the threshold {resp.confidence_threshold:.2f}._"
    elif resp.mode != "rag":
        answer = f"_{answer}_"
    if resp.rewritten_query != question:
        answer += f"\n\n<sub>Follow-up interpreted as: “{html.escape(resp.rewritten_query)}”</sub>"
    lat = " | ".join(f"{k.replace('_ms', '')}: {v:.0f} ms" for k, v in resp.latency_ms.items())
    lat = f"<sub>{LLM_STATUS} — confidence {resp.confidence:.2f} — {lat}</sub>"
    rows = [[r.segment_id, r.title, r.timestamp, round(r.retrieval_score, 3),
             None if r.rerank_score is None else round(r.rerank_score, 2), r.snippet] for r in resp.retrieved]
    history = (history + [{"role": "user", "content": question},
                          {"role": "assistant", "content": resp.answer}])[-6:]
    return answer, render_citations(resp), lat, gr.update(value=rows, visible=show_retrieved), history


with gr.Blocks(title="ConceptBound: ask your lectures") as demo:
    gr.Markdown("# ConceptBound\nAsk a question about the indexed lectures. Answers cite the exact moments "
                "they come from; if the lectures don't cover it, the system says so instead of guessing.")
    history = gr.State([])
    with gr.Row():
        q = gr.Textbox(label="Your question", placeholder="e.g. Explain gradient descent from the lecture", scale=5)
        btn = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=q)
    with gr.Row():
        use_rr = gr.Checkbox(value=True, label="Use cross-encoder reranker")
        show = gr.Checkbox(value=False, label="Show retrieved segments")
        clear = gr.Button("New conversation", size="sm")
    answer = gr.Markdown()
    latency = gr.Markdown()
    cites = gr.HTML()
    table = gr.Dataframe(headers=["segment", "lecture", "start", "retrieval score", "rerank score", "snippet"],
                         visible=False, wrap=True, label="Retrieved segments")
    gr.Markdown(f"<sub>{LLM_STATUS}. Index: {len(SEARCH.retriever.metadata)} segments.</sub>")

    outs = [answer, cites, latency, table, history]
    btn.click(ask, [q, history, use_rr, show], outs)
    q.submit(ask, [q, history, use_rr, show], outs)
    show.change(lambda v: gr.update(visible=v), show, table)
    clear.click(lambda: ([], "", "", "", gr.update(value=[])), None, [history, answer, cites, latency, table])

if __name__ == "__main__":
    demo.launch(allowed_paths=[VIDEOS_DIR] if os.path.isdir(VIDEOS_DIR) else None)
