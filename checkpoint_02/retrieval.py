"""BM25 with positive log1p IDF; no automatic results for all-zero scores."""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import math

import numpy as np
from nltk.stem.snowball import RussianStemmer

from .corpus import TOKEN_RE
from .splitters import Chunk

_STEMMER = RussianStemmer()


@lru_cache(maxsize=50000)
def stem(word: str) -> str:
    return _STEMMER.stem(word)


def tokenize(text: str, mode: str = "lower") -> list[str]:
    if mode not in {"lower", "stem"}:
        raise ValueError("Unknown tokenization mode")
    words = TOKEN_RE.findall(text.lower().replace("\u0451", "\u0435"))
    return [stem(w) for w in words] if mode == "stem" else words


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75, mode: str = "lower"):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and b in [0, 1]")
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs")
        self.chunks, self.k1, self.b, self.mode = chunks, k1, b, mode
        counts = [Counter(tokenize(c.text, mode)) for c in chunks]
        lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        if lengths.sum() == 0:
            raise ValueError("Corpus contains no tokens")
        self.lengths = lengths
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        temporary: dict[str, list[tuple[int, int]]] = {}
        for i, counter in enumerate(counts):
            for term, count in counter.items():
                temporary.setdefault(term, []).append((i, count))
        self.idf = {}
        for term, rows in temporary.items():
            self.postings[term] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=float))
            df = len(rows)
            self.idf[term] = math.log1p((len(chunks) - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for term in set(tokenize(query, self.mode)):
            if term in self.postings:
                indices, frequencies = self.postings[term]
                scores[indices] += self.idf[term] * frequencies * (self.k1 + 1) / (frequencies + self.norm[indices])
        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = self.scores(query)
        ranked = sorted(np.flatnonzero(scores > 0), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [(self.chunks[i], float(scores[i])) for i in ranked[:top_k]]
