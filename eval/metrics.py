"""
Ranking metrics with binary relevance (a segment is relevant if it is gold).

  Recall@k  fraction of a question's gold segments found in the top k
  MRR       1 / rank of the first gold segment (0 if none in the list)
  nDCG@10   DCG of the ranking divided by the best possible DCG, with
            gain 1 for gold segments and log2(rank+1) discounting
"""

import math
from typing import Dict, List, Sequence, Set

import numpy as np


def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def mrr(ranked: Sequence[str], gold: Set[str]) -> float:
    for i, s in enumerate(ranked, start=1):
        if s in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], start=1) if s in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(ranked: Sequence[str], gold: Set[str]) -> Dict[str, float]:
    return {"R@5": recall_at_k(ranked, gold, 5), "R@10": recall_at_k(ranked, gold, 10),
            "MRR": mrr(ranked, gold), "nDCG@10": ndcg_at_k(ranked, gold, 10)}


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def percentiles(ms: Sequence[float]) -> Dict[str, float]:
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95))} if len(ms) else {}
