"""
FastAPI backend for the Concept-Aware Lecture Retrieval demo.

Endpoints
---------
GET  /api/health       — system health + diagnostics
GET  /api/search?q=<query>&top_k=<k>  — search the lecture corpus
GET  /api/stats        — index statistics
POST /ask              — grounded answer with timestamp citations (RAG)
GET  /health           — liveness + whether the RAG index and LLM are ready

Run with:
    uvicorn app.backend.main:app --reload --port 8000

The frontend (app/frontend/index.html) is also served as a static page at
"/" for convenience.
"""

import os
import traceback

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import PATHS
from src.stage2_retrieval.retrieval import LectureRetriever
from src.rag.schema import AskRequest, AskResponse

app = FastAPI(title="Concept-Aware Lecture Video Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever = None
_retriever_error = None


def get_retriever() -> LectureRetriever:
    global _retriever, _retriever_error
    if _retriever is None and _retriever_error is None:
        try:
            _retriever = LectureRetriever(index_dir=PATHS.index_dir)
        except Exception as e:
            _retriever_error = str(e)
            print(f"[API] Failed to load retriever: {e}")
            traceback.print_exc()
    return _retriever


@app.get("/api/health")
def health():
    index_exists = os.path.exists(os.path.join(PATHS.index_dir, "segments.index"))
    metadata_exists = os.path.exists(os.path.join(PATHS.index_dir, "segments_metadata.json"))
    stage1_exists = os.path.exists(os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    stage2_exists = os.path.exists(os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))

    retriever = get_retriever()
    num_segments = 0
    if retriever is not None:
        num_segments = retriever.index.ntotal

    return {
        "status": "ok" if index_exists else "index_missing",
        "index_ready": index_exists and metadata_exists,
        "num_segments": num_segments,
        "stage1_model_ready": stage1_exists,
        "stage2_model_ready": stage2_exists,
        "retriever_loaded": retriever is not None,
        "retriever_error": _retriever_error,
    }


@app.get("/api/stats")
def stats():
    retriever = get_retriever()
    if retriever is None:
        return {"error": _retriever_error or "Retriever not loaded. Build the index first."}

    # Count unique videos
    video_ids = set()
    for meta in retriever.metadata:
        video_ids.add(meta["video_id"])

    return {
        "num_segments": retriever.index.ntotal,
        "num_videos": len(video_ids),
        "video_ids": sorted(video_ids),
        "has_model": retriever._model is not None,
    }


@app.get("/api/search")
def search(q: str = Query(..., description="Free-text concept query"),
           top_k: int = Query(5, ge=1, le=20)):
    retriever = get_retriever()
    if retriever is None:
        return {
            "query": q,
            "results": [],
            "error": _retriever_error or "Index not built. Run the pipeline first.",
        }
    try:
        # Pass a URL template that points to the mounted raw videos directory
        results = retriever.search(q, top_k=top_k, video_url_template="/videos/{video_id}.mp4#t={start}")
        return {"query": q, "results": results}
    except Exception as e:
        traceback.print_exc()
        return {"query": q, "results": [], "error": str(e)}


# ---- RAG ---------------------------------------------------------------------------
_rag = None
_rag_error = None


def get_rag():
    """Build the RAG pipeline once: precomputed index + reranker + LLM from env."""
    global _rag, _rag_error
    if _rag is None and _rag_error is None:
        try:
            from src.llm.factory import get_llm_client
            from src.rag.pipeline import RAGPipeline
            from src.retrieval.search import SearchService
            search = SearchService(use_reranker=os.environ.get("RAG_USE_RERANKER", "1") == "1")
            search.warm_up()
            _rag = RAGPipeline(search, get_llm_client())
        except Exception as e:
            _rag_error = str(e)
            traceback.print_exc()
    return _rag


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail=f"RAG pipeline not available: {_rag_error}. "
                                                    f"Build it with: python -m src.rag.build_index")
    return rag.ask(req.question, history=req.history, use_reranker=req.use_reranker, top_k=req.top_k)


@app.get("/health")
def rag_health():
    rag = get_rag()
    return {
        "status": "ok" if rag is not None else "degraded",
        "rag_index_loaded": rag is not None,
        "num_segments": len(rag.search.retriever.metadata) if rag else 0,
        "reranker": rag is not None and rag.search.reranker is not None,
        "llm": f"{rag.llm.name}:{rag.llm.model}" if rag and rag.llm else None,
        "mode": ("rag" if rag and rag.llm else "retrieval_only") if rag else None,
        "error": _rag_error,
    }


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Mount the raw videos directory so links to /videos/...mp4 will play in the browser
if os.path.isdir(PATHS.raw_videos_dir):
    app.mount("/videos", StaticFiles(directory=PATHS.raw_videos_dir), name="videos")
