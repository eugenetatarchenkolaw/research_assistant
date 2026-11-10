"""Macro retrieval metrics; empty qrels remain in the denominator."""
from __future__ import annotations
from math import log2
from typing import Mapping, Sequence
import numpy as np


def per_query_metrics(ranked: Sequence[str], relevant: Sequence[str], k: int) -> dict[str, float]:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Repeated result IDs would inflate retrieval metrics")
    gold = set(relevant)
    hits = [int(cid in gold) for cid in ranked[:k]]
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dict(hit=float(bool(sum(hits))), precision=sum(hits) / k,
                recall=sum(hits) / len(gold) if gold else 0.0,
                mrr=1 / first if first else 0.0, ndcg=dcg / ideal if ideal else 0.0,
                addressable=float(bool(gold)))


def evaluate_retrieval(retrieved: Mapping[str, Sequence[str]], relevant: Mapping[str, Sequence[str]], k: int) -> dict:
    if not relevant:
        raise ValueError("Evaluation requires questions, not an empty qrels mapping")
    if set(retrieved) - set(relevant):
        raise ValueError("Predictions contain unknown question IDs")
    rows = [per_query_metrics(retrieved.get(qid, []), gold, k) for qid, gold in relevant.items()]
    return {**{name: float(np.mean([r[name] for r in rows])) for name in rows[0]}, "questions": len(rows)}


def paired_group_interval(a: Sequence[float], b: Sequence[float], groups: Sequence[str], seed: int = 42, draws: int = 2000) -> tuple[float, float]:
    """Paired cluster bootstrap by source page; intervals are exploratory."""
    if len(a) != len(b) or len(a) != len(groups) or not a:
        raise ValueError("Nonempty arrays of equal length are required")
    keys = sorted(set(groups))
    differences = np.asarray(a) - np.asarray(b)
    indices = [np.flatnonzero(np.asarray(groups) == key) for key in keys]
    rng = np.random.default_rng(seed)
    samples = [differences[np.concatenate([indices[i] for i in rng.integers(0, len(keys), len(keys))])].mean() for _ in range(draws)]
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))
