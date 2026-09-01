"""
Query-time retrieval: encode a free-text student query, search the FAISS
index of segment embeddings, and return the top-k ranked
(video_id, start_time, end_time, ...) results with "jump to moment" links.

When the FAISS model returns low-confidence results (which happens when the
Stage 2 model hasn't been well-trained), a fallback text-similarity search
is used: direct Sentence-BERT cosine similarity between the query and each
segment's stored transcript/OCR text.  The two scores are combined for the
final ranking, ensuring search works reasonably even before the model is
properly trained.
"""

import os
from typing import List, Dict

import numpy as np
import torch

from config import PATHS
from src.stage2_retrieval.encoders import TextEncoder
from src.stage2_retrieval.index_builder import load_stage2_model
from src.utils.io_utils import load_json, format_timestamp


class LectureRetriever:
    def __init__(self, index_dir: str = PATHS.index_dir,
                 checkpoint_path: str = None,
                 device: str = "cpu"):
        import faiss
        import time

        checkpoint_path = checkpoint_path or os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")

        self.device = device
        self.text_encoder = TextEncoder(device=device)

        # Load FAISS index and metadata
        self.index = faiss.read_index(os.path.join(index_dir, "segments.index"))
        self.metadata: List[Dict] = load_json(os.path.join(index_dir, "segments_metadata.json"))

        # Precompute fallback text embeddings at startup for sub-second search latency
        t0 = time.time()
        print(f"[LectureRetriever] Pre-computing fallback text embeddings for {len(self.metadata)} segments...")
        
        texts_to_encode = []
        for meta in self.metadata:
            text = (meta.get("transcript_text", "") + " " + meta.get("ocr_text", "")).strip()
            texts_to_encode.append(text if text else " ")
            
        if texts_to_encode:
            embeds = self.text_encoder.encode(texts_to_encode)
            self.segment_text_embeddings = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
        else:
            self.segment_text_embeddings = np.zeros((0, self.text_encoder.dim), dtype=np.float32)
            
        print(f"[LectureRetriever] Text embeddings pre-computed in {time.time() - t0:.2f}s.")

        # Build BM25 keyword index for deterministic fallback search
        self._build_bm25_index()

        # Try to load the trained Stage 2 model; if missing, use text-only search
        self._model = None
        if os.path.exists(checkpoint_path):
            try:
                self._model = load_stage2_model(checkpoint_path, device)
            except Exception as e:
                print(f"[LectureRetriever] Could not load Stage 2 model: {e}")
                print("[LectureRetriever] Falling back to text-similarity search.")

    def _build_bm25_index(self):
        """Build a BM25 keyword index over segment texts for deterministic fallback.

        BM25 (Best Matching 25) is a proven keyword-based ranking function that
        guarantees accurate results even when the neural model collapses. Unlike
        Sentence-BERT semantic search, BM25 excels at exact keyword matching
        (e.g., student searches "merge sort" and transcript says "merge sort").
        """
        try:
            from rank_bm25 import BM25Okapi

            corpus = []
            for meta in self.metadata:
                text = (meta.get("transcript_text", "") + " " +
                        meta.get("ocr_text", "")).strip().lower()
                corpus.append(text.split() if text else ["."])
            self.bm25 = BM25Okapi(corpus)
            print(f"[LectureRetriever] BM25 index built over {len(corpus)} segments.")
        except ImportError:
            print("[LectureRetriever] rank-bm25 not installed, BM25 fallback disabled.")
            self.bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 keyword-based search as a deterministic fallback."""
        if self.bm25 is None:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            meta = self.metadata[idx]
            results.append({
                "video_id": meta["video_id"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "start_time_str": format_timestamp(meta["start_time"]),
                "end_time_str": format_timestamp(meta["end_time"]),
                "ocr_text": meta.get("ocr_text", ""),
                "transcript_text": meta.get("transcript_text", ""),
                "score": float(scores[idx]),
                "search_mode": "bm25",
            })
        return results

    def _text_similarity_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback: rank segments by direct Sentence-BERT cosine similarity
        between the query and each segment's transcript + OCR text."""
        if len(self.segment_text_embeddings) == 0:
            return []
            
        query_emb = self.text_encoder.encode([query])[0]  # (D,)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Vectorized dot product against precomputed embeddings (extremely fast)
        scores = np.dot(self.segment_text_embeddings, query_emb)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append({
                "video_id": meta["video_id"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "start_time_str": format_timestamp(meta["start_time"]),
                "end_time_str": format_timestamp(meta["end_time"]),
                "ocr_text": meta.get("ocr_text", ""),
                "transcript_text": meta.get("transcript_text", ""),
                "score": float(scores[idx]),
                "search_mode": "text_similarity",
            })
        return results

    def _combined_fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Hybrid BM25 + semantic fallback: combines keyword and semantic scores.

        Uses a 50/50 weighted combination of BM25 keyword scores (exact match)
        and Sentence-BERT cosine similarity (semantic match) for robust results.
        """
        text_results = self._text_similarity_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        # Build score lookup by segment key
        combined = {}
        for r in text_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            combined[key] = {**r, "text_score": r["score"], "bm25_score": 0.0}

        # Normalize BM25 scores to [0, 1] range
        bm25_scores = [r["score"] for r in bm25_results] if bm25_results else [0.0]
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for r in bm25_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            norm_score = r["score"] / bm25_max
            if key in combined:
                combined[key]["bm25_score"] = norm_score
            else:
                combined[key] = {**r, "text_score": 0.0, "bm25_score": norm_score}

        # Combine: 50% semantic + 50% keyword
        for key, r in combined.items():
            r["score"] = 0.5 * r["text_score"] + 0.5 * r["bm25_score"]
            r["search_mode"] = "bm25_fallback"

        merged = sorted(combined.values(), key=lambda x: -x["score"])
        return merged[:top_k]

    def search(self, query: str, top_k: int = 5,
               video_url_template: str = "{video_id}#t={start}") -> List[Dict]:
        top_k = min(top_k, max(1, self.index.ntotal))

        # --- FAISS model-based search ---
        faiss_results = []
        if self._model is not None:
            query_emb = self.text_encoder.encode([query])  # (1, text_embed_dim)
            query_tensor = torch.from_numpy(query_emb.astype(np.float32)).to(self.device)

            with torch.no_grad():
                query_z = self._model.encode_queries(query_tensor)
            query_z_np = query_z.cpu().numpy().astype(np.float32)

            scores, indices = self.index.search(query_z_np, top_k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                meta = self.metadata[idx]
                faiss_results.append({
                    "video_id": meta["video_id"],
                    "start_time": meta["start_time"],
                    "end_time": meta["end_time"],
                    "start_time_str": format_timestamp(meta["start_time"]),
                    "end_time_str": format_timestamp(meta["end_time"]),
                    "ocr_text": meta.get("ocr_text", ""),
                    "transcript_text": meta.get("transcript_text", ""),
                    "score": float(score),
                    "search_mode": "model",
                })

        # --- Fallback search (BM25 + semantic hybrid) ---
        if not faiss_results:
            return self._finalize_results(
                self._combined_fallback_search(query, top_k), video_url_template)

        scores_arr = [r["score"] for r in faiss_results]
        avg_faiss_score = np.mean(scores_arr)
        variance = np.var(scores_arr)

        # Dynamic variance trigger: if the model is untrained (low scores),
        # collapsed (all scores identical / very high), bypass neural results
        # and use the deterministic BM25 + semantic hybrid fallback.
        if avg_faiss_score < 0.15 or avg_faiss_score > 0.85 or variance < 0.01:
            # Model is untrained or collapsed — use BM25 + semantic hybrid
            return self._finalize_results(
                self._combined_fallback_search(query, top_k), video_url_template)

        # Merge: model results are primary, boosted by text similarity
        text_results = self._text_similarity_search(query, top_k)

        faiss_lookup = {}
        for r in faiss_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            faiss_lookup[key] = r

        text_lookup = {}
        for r in text_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            text_lookup[key] = r["score"]

        # Boost FAISS scores with text similarity
        for key, r in faiss_lookup.items():
            text_boost = text_lookup.get(key, 0.0)
            r["score"] = 0.7 * r["score"] + 0.3 * text_boost
            r["search_mode"] = "hybrid"

        merged = sorted(faiss_lookup.values(), key=lambda x: -x["score"])
        return self._finalize_results(merged[:top_k], video_url_template)

    def _finalize_results(self, results: List[Dict], video_url_template: str) -> List[Dict]:
        for r in results:
            r["jump_link"] = video_url_template.format(
                video_id=r["video_id"], start=int(r["start_time"])
            )
        return results
