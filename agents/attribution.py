"""
Source-weighting / traceability: map answer sentences to retrieval chunks.

Each sentence gets the best-matching chunk by lexical overlap (no extra model load).
Sentences with weak support are flagged ``speculative`` for UI disclosure.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

# Below this best Jaccard(word) score, label sentence as speculative
DEFAULT_SPECULATIVE_THRESHOLD = 0.12

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def attribute_answer_to_chunks(
    answer: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    speculative_threshold: float = DEFAULT_SPECULATIVE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    For each non-trivial sentence, find the chunk with highest token Jaccard overlap.

    Returns dicts with: sentence_index, sentence, best_source_index (1-based),
    best_score, source, speculative.
    """
    if not answer.strip() or not chunks:
        return []

    chunk_texts = [(i, (c.get("text") or "")) for i, c in enumerate(chunks)]
    chunk_sets = [(i, _tokens(t)) for i, t in chunk_texts]

    raw_parts = [p.strip() for p in _SENT_SPLIT.split(answer) if p.strip()]
    out: List[Dict[str, Any]] = []
    idx = 0
    for sent in raw_parts:
        if len(sent) < 12:
            continue
        st = _tokens(sent)
        if len(st) < 2:
            continue
        best_i = 0
        best_score = 0.0
        for ci, cset in chunk_sets:
            s = _jaccard(st, cset)
            if s > best_score:
                best_score = s
                best_i = ci
        src = chunks[best_i].get("source", f"chunk_{best_i + 1}")
        speculative = best_score < speculative_threshold
        aid = chunks[best_i].get("attribution_id")
        out.append(
            {
                "sentence_index": idx,
                "sentence": sent[:2000],
                "best_source_index": best_i + 1,
                "best_score": round(best_score, 4),
                "source": src,
                "speculative": speculative,
                "attribution_id": aid,
            }
        )
        idx += 1

    return out
