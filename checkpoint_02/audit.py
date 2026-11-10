"""Duplicate candidates and corpus summaries; no automatic fuzzy deletion."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from rapidfuzz.distance import Levenshtein

from .corpus import normalize
from .retrieval import tokenize

BASIC_STOP = set("\u0438 \u0432 \u0432\u043e \u043d\u0430 \u043f\u043e \u043a \u043a\u043e \u0441 \u0441\u043e \u043e \u043e\u0431 \u043e\u0442 \u0434\u043e \u0438\u0437 \u0437\u0430 \u0434\u043b\u044f \u043f\u0440\u0438 \u043d\u043e \u0430 \u0438\u043b\u0438 \u043a\u0430\u043a \u0442\u0430\u043a \u0447\u0442\u043e \u044d\u0442\u043e \u0435\u0435 \u0435\u0433\u043e \u0438\u0445 \u043e\u043d\u0438 \u043e\u043d\u0430 \u043e\u043d \u044d\u0442\u043e\u0442 \u044d\u0442\u043e\u0439 \u044d\u0442\u043e\u0433\u043e \u044d\u0442\u0438\u0445 \u044d\u0442\u0438 \u043a\u043e\u0442\u043e\u0440\u044b\u0435 \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u043a\u043e\u0442\u043e\u0440\u044b\u0445".split())
DOMAIN_STOP = set("\u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0446\u0438\u044f \u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0446\u0438\u0438 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u043e \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u044b \u0441\u0438\u0441\u0442\u0435\u043c\u0430 \u043c\u0435\u043d\u0435\u0434\u0436\u043c\u0435\u043d\u0442\u0430 \u0434\u043e\u043b\u0436\u043d\u0430 \u0434\u043e\u043b\u0436\u0435\u043d \u0434\u043e\u043b\u0436\u043d\u044b \u0441\u043b\u0435\u0434\u0443\u0435\u0442 \u0438\u0441\u043e \u0441\u0442\u0430\u043d\u0434\u0430\u0440\u0442\u0430".split())


def word_frequencies(text: str, stop_mode: str = "basic") -> Counter:
    if stop_mode not in {"none", "basic", "domain"}:
        raise ValueError("Unknown stop-word mode")
    stops = set() if stop_mode == "none" else BASIC_STOP
    if stop_mode == "domain":
        stops = stops | DOMAIN_STOP
    return Counter(w for w in tokenize(text) if len(w) > 1 and not any(c.isdigit() for c in w) and w not in stops)


def duplicate_candidates(items: list[tuple[str, str]], threshold: float = 0.85) -> list[dict]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("Duplicate input IDs")
    normalized = [(key, normalize(text)) for key, text in items if normalize(text)]
    result = []
    for (key_a, a), (key_b, b) in combinations(normalized, 2):
        # The length bound is exact: a shorter string cannot exceed this score.
        if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
            continue
        exact = sha256(a.encode()).digest() == sha256(b.encode()).digest()
        score = 1.0 if exact else Levenshtein.normalized_similarity(a, b, score_cutoff=threshold)
        if score >= threshold:
            result.append(dict(left=key_a, right=key_b, similarity=float(score), method="sha256" if exact else "levenshtein"))
    return sorted(result, key=lambda r: (-r["similarity"], r["left"], r["right"]))
