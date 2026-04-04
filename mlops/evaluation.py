from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    expected: str


def retrieval_hit_rate(samples: List[EvalSample]) -> float:
    if not samples:
        return 0.0
    hits = 0
    for s in samples:
        expected_terms = set(s.expected.lower().split())
        context_blob = ' '.join(s.contexts).lower()
        if expected_terms and any(t in context_blob for t in expected_terms):
            hits += 1
    return hits / len(samples)
