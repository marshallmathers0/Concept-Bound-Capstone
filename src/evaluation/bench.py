"""
Evaluation harness: segmenters x retrievers, scored on human gold data only.

Segmenters (how a lecture is cut into retrievable units)
    stage1        trained Pedagogical Boundary Detector
    pseudo        the peak-picking heuristic Stage 1 is (pre)trained on. If
                  stage1 does not beat this, it has only learned the heuristic.
    topic_drift   TextTiling-style: peaks of transcript topic drift only
    shot          visual-change peaks (generic shot detection)
    fixed_<N>     uniform N-second windows
    gold          the annotators' own segmentation (oracle upper bound for
                  "does better segmentation help retrieval?")

Retrievers (how a query is matched to units)
    bm25          BM25 over transcript + slide text
    dense:<name>  off-the-shelf sentence embedding model (default MiniLM, BGE)
    hybrid        reciprocal-rank fusion of bm25 and the first dense model
    fusion        the trained Stage-2 cross-modal model
    +rerank       optional cross-encoder re-ranking of the hybrid top-20

All segmenters are restricted to the gold `lecture_span`, and all retrievers
see identical text for a unit (src/utils/segment_text.py).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from config import MODELS, PATHS, STAGE1
from src.baselines.fixed_window import fixed_window_segments
from src.evaluation.boundary_metrics import evaluate_boundaries_full
from src.feature_extraction.feature_pipeline import load_feature_stream
from src.stage1_boundary.dataset import span_mask
from src.stage1_boundary.pseudo_labels import (
    boundaries_from_annotations, pseudo_boundaries_from_signals, smooth_signal,
)
from src.stage1_boundary.segment import boundaries_to_segments, enforce_segment_lengths
from src.utils.gold import load_gold
from src.utils.segment_text import span_ocr, span_transcript
from src.utils.text_search import bm25_tokenize


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------

def _length_steps(timestamps):
    step = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    return (max(1, int(round(STAGE1.min_segment_sec / step))),
            max(2, int(round(STAGE1.max_segment_sec / step))))


def _segments_from_signal(score: np.ndarray, candidates: np.ndarray, timestamps) -> List[Dict]:
    """Apply the same length constraints Stage 1 uses, so baselines are not
    handicapped by producing 9-second segments."""
    min_steps, max_steps = _length_steps(timestamps)
    idx = enforce_segment_lengths(candidates, score, min_steps, max_steps)
    return boundaries_to_segments(idx, timestamps, min_segment_len_steps=1)


def _norm(x):
    x = smooth_signal(x.astype(np.float32), STAGE1.peak_smoothing_window)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-8 else np.zeros_like(x)


class Segmenters:
    def __init__(self, stage1_checkpoint: Optional[str], device: str = "cpu"):
        self.device = device
        self.stage1 = None
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            from src.stage1_boundary.inference import load_stage1_model
            self.stage1 = load_stage1_model(stage1_checkpoint, device)

    def names(self, requested: List[str]) -> List[str]:
        out = []
        for n in requested:
            if n == "stage1" and self.stage1 is None:
                print("[bench] no Stage 1 checkpoint, skipping 'stage1'")
                continue
            out.append(n)
        return out

    def segment(self, name: str, feats: Dict, keep: np.ndarray, gold: Dict) -> List[Dict]:
        ts = feats["timestamps"][keep].astype(np.float32)
        if name.startswith("fixed_"):
            return fixed_window_segments(ts, window_sec=float(name.split("_")[1]))
        if name == "gold":
            gt = boundaries_from_annotations(ts, gold["boundaries_sec"])
            return boundaries_to_segments(np.where(gt == 1)[0], ts, min_segment_len_steps=1)
        if name == "stage1":
            x = torch.from_numpy(np.ascontiguousarray(feats["features"][keep])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = self.stage1.boundary_scores(self.stage1(x))[0].cpu().numpy()
            from src.stage1_boundary.segment import segments_from_scores
            return segments_from_scores(scores, ts)

        ocr = feats["ocr_signal"][keep]
        drift = feats["topic_drift_signal"][keep]
        vis = feats["visual_signal"][keep]
        if name == "pseudo":
            cand = np.where(pseudo_boundaries_from_signals(ocr, drift, vis) == 1)[0]
            score = (0.7 * (0.5 * _norm(ocr) + 0.5 * _norm(drift)) + 0.3 * _norm(vis))[1:]
            return _segments_from_signal(score, cand, ts)
        if name == "topic_drift":
            score = _norm(drift)[1:]
            cand = np.where(score > score.mean() + score.std())[0]
            return _segments_from_signal(score, cand, ts)
        if name == "shot":
            score = _norm(vis)[1:]
            cand = np.where(score > score.mean() + 1.5 * score.std())[0]
            return _segments_from_signal(score, cand, ts)
        raise ValueError(f"unknown segmenter {name}")


def evaluate_stage1(video_ids: List[str], segmenter_names: List[str], segs: Segmenters,
                    features_dir: str = PATHS.features_dir) -> Dict:
    """Per-video boundary metrics vs the primary annotator, averaged."""
    per_method = {n: [] for n in segmenter_names if n != "gold"}
    for vid in video_ids:
        gold = load_gold(vid)
        if gold is None or not gold["boundaries_sec"]:
            continue
        feats = load_feature_stream(os.path.join(features_dir, f"{vid}.npz"))
        keep = span_mask(feats["timestamps"], gold)
        ts = feats["timestamps"][keep]
        gt_segments = segs.segment("gold", feats, keep, gold)
        for name in per_method:
            pred = segs.segment(name, feats, keep, gold)
            per_method[name].append({"video_id": vid, **evaluate_boundaries_full(pred, gt_segments, len(ts))})
    summary = {}
    for name, rows in per_method.items():
        if not rows:
            continue
        keys = [k for k in rows[0] if k != "video_id"]
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        summary[name]["n_videos"] = len(rows)
    return {"summary": summary, "per_video": per_method}


# ---------------------------------------------------------------------------
# Corpus + retrievers
# ---------------------------------------------------------------------------

def build_corpus(video_ids: List[str], segmenter: str, segs: Segmenters,
                 features_dir: str = PATHS.features_dir) -> List[Dict]:
    units = []
    for vid in video_ids:
        gold = load_gold(vid)
        feats = load_feature_stream(os.path.join(features_dir, f"{vid}.npz"))
        keep = span_mask(feats["timestamps"], gold)
        for seg in segs.segment(segmenter, feats, keep, gold or {"boundaries_sec": []}):
            st, et = float(seg["start_time"]), float(seg["end_time"])
            units.append({"video_id": vid, "start_time": st, "end_time": et,
                          "transcript": span_transcript(feats, st, et),
                          "ocr": span_ocr(feats, st, et)})
    return units


def _rank(scores: np.ndarray, units: List[Dict], top_k: int, mode: str) -> List[Dict]:
    order = np.argsort(-scores)[:top_k]
    return [{"video_id": units[i]["video_id"], "start_time": units[i]["start_time"],
             "end_time": units[i]["end_time"], "score": float(scores[i]), "search_mode": mode}
            for i in order]


class BM25Retriever:
    name = "bm25"

    def __init__(self, units):
        from rank_bm25 import BM25Okapi
        self.units = units
        self.bm25 = BM25Okapi([bm25_tokenize(u["transcript"] + " " + u["ocr"]) or ["."] for u in units])

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(bm25_tokenize(query)), dtype=np.float64)

    def search(self, query, top_k):
        return _rank(self.scores(query), self.units, top_k, self.name)


class DenseRetriever:
    """Off-the-shelf sentence embeddings over unit text (chunked + averaged)."""

    QUERY_PREFIX = {"BAAI/bge": "Represent this sentence for searching relevant passages: ",
                    "intfloat/e5": "query: "}

    def __init__(self, units, model_name: str, device: str = "cpu"):
        from src.feature_extraction.asr_extractor import SentenceEncoder
        self.name = f"dense:{model_name.split('/')[-1]}"
        self.units = units
        self.enc = SentenceEncoder(model_name, device=device)
        self.prefix = next((p for k, p in self.QUERY_PREFIX.items() if model_name.startswith(k)), "")
        doc_prefix = "passage: " if model_name.startswith("intfloat/e5") else ""
        docs = [doc_prefix + (u["transcript"] + " " + u["ocr"]).strip() for u in units]
        emb = self.enc.encode_long(docs)
        self.emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    def scores(self, query):
        q = self.enc.encode([self.prefix + query])[0]
        return self.emb @ (q / (np.linalg.norm(q) + 1e-8))

    def search(self, query, top_k):
        return _rank(self.scores(query), self.units, top_k, self.name)


class HybridRetriever:
    """Reciprocal rank fusion (k=60) of two retrievers - robust to their
    different score scales, unlike a weighted sum of raw scores."""

    def __init__(self, a, b, k: int = 60):
        self.a, self.b, self.k = a, b, k
        self.units = a.units
        self.name = f"hybrid({a.name}+{b.name})"

    def scores(self, query):
        out = np.zeros(len(self.units))
        for r in (self.a, self.b):
            ranks = np.empty(len(self.units))
            ranks[np.argsort(-r.scores(query))] = np.arange(1, len(self.units) + 1)
            out += 1.0 / (self.k + ranks)
        return out

    def search(self, query, top_k):
        return _rank(self.scores(query), self.units, top_k, self.name)


class RerankRetriever:
    def __init__(self, first_stage, model_name: str = MODELS.cross_encoder, depth: int = 20, device="cpu"):
        from sentence_transformers import CrossEncoder
        self.first, self.depth = first_stage, depth
        self.units = first_stage.units
        self.ce = CrossEncoder(model_name, device=device, max_length=512)
        self.name = f"{first_stage.name}+rerank"

    def search(self, query, top_k):
        base = self.first.scores(query)
        cand = np.argsort(-base)[:self.depth]
        pairs = [(query, (self.units[i]["transcript"] + " " + self.units[i]["ocr"])[:2000]) for i in cand]
        ce = np.asarray(self.ce.predict(pairs, show_progress_bar=False), dtype=np.float64)
        scores = np.full(len(self.units), -np.inf)
        scores[cand] = ce
        return _rank(scores, self.units, top_k, self.name)


class FusionRetriever:
    """The trained Stage-2 model applied to this corpus' units."""

    def __init__(self, units, checkpoint: str, device: str = "cpu",
                 features_dir: str = PATHS.features_dir):
        from src.stage2_retrieval.encoders import TextEncoder
        from src.stage2_retrieval.index_builder import load_stage2_model
        from src.stage2_retrieval.segment_features import segment_bundle

        self.name = "fusion"
        self.units, self.device = units, device
        self.model = load_stage2_model(checkpoint, device)
        self.text = TextEncoder(device=device)
        z = []
        by_video: Dict[str, List[int]] = {}
        for i, u in enumerate(units):
            by_video.setdefault(u["video_id"], []).append(i)
        z = np.zeros((len(units), self.model.segment_encoder.out_proj[-1].out_features), dtype=np.float32)
        for vid, idx in by_video.items():
            feats = load_feature_stream(os.path.join(features_dir, f"{vid}.npz"))
            b = segment_bundle(feats, [units[i] for i in idx], self.text)
            with torch.no_grad():
                zz = self.model.encode_segments(*(torch.from_numpy(b[k].astype(np.float32)).to(device)
                                                  for k in ("ocr_emb", "transcript_emb", "visual_emb")))
            z[idx] = zz.cpu().numpy()
        self.z = z

    def scores(self, query):
        q = torch.from_numpy(self.text.encode([query]).astype(np.float32)).to(self.device)
        with torch.no_grad():
            qz = self.model.encode_queries(q).cpu().numpy()[0]
        return self.z @ qz

    def search(self, query, top_k):
        return _rank(self.scores(query), self.units, top_k, self.name)


def build_retrievers(units, dense_models: List[str], stage2_checkpoint: Optional[str],
                     rerank: bool, device: str) -> List:
    rs = [BM25Retriever(units)]
    dense = [DenseRetriever(units, m, device) for m in dense_models]
    rs += dense
    if dense:
        rs.append(HybridRetriever(rs[0], dense[0]))
        if rerank:
            rs.append(RerankRetriever(rs[-1], device=device))
    if stage2_checkpoint and os.path.exists(stage2_checkpoint):
        rs.append(FusionRetriever(units, stage2_checkpoint, device))
    return rs


def gold_queries(video_ids: List[str]) -> List[Dict]:
    out = []
    for vid in video_ids:
        gold = load_gold(vid)
        if gold is None:
            continue
        for q in gold["queries"]:
            out.append({"video_id": vid, "query": q["query"], "start_time": float(q["start_time"]),
                        "end_time": float(q["end_time"]), "type": q.get("type", "other"),
                        "annotator": q.get("annotator")})
    return out
