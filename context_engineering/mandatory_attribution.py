"""
Mandatory attribution IDs and grounding confidence for RAG outputs.

Assigns stable ``attribution_id`` values to retrieved chunks so UI / logs can link
claims back to ingestion. ``grounding_confidence`` estimates how much of the
answer is supported by retrieved text (vs. likely parametric knowledge).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

from agents.attribution import attribute_answer_to_chunks


def enrich_chunks_with_attribution_ids(
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return new chunk dicts with ``attribution_id`` (stable given source index + text prefix)."""
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        src = str(c.get("source", f"chunk_{i}"))
        text = (c.get("text") or "")[:200]
        h = hashlib.sha256(f"{src}|{i}|{text}".encode("utf-8")).hexdigest()[:16]
        cid = f"SRC-{h.upper()}"
        out.append({**dict(c), "attribution_id": cid, "source_index": i + 1})
    return out


def compute_grounding_confidence(answer: str, chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Fraction of substantive sentences that are not flagged speculative by lexical
    overlap (see :mod:`agents.attribution`). Returns 1.0 for empty answers.
    """
    if not (answer or "").strip():
        return 1.0
    spans = attribute_answer_to_chunks(answer, list(chunks))
    if not spans:
        return 0.35
    grounded = sum(1 for s in spans if not s.get("speculative"))
    return round(grounded / len(spans), 4)


def build_attribution_footer(chunks: Sequence[Dict[str, Any]], max_lines: int = 12) -> str:
    """Human-readable map attribution_id → source for prompts or UI."""
    lines = ["[Source attribution map]"]
    for i, c in enumerate(chunks[:max_lines]):
        aid = c.get("attribution_id", "?")
        src = c.get("source", f"doc_{i + 1}")
        lines.append(f"  {aid} → {src}")
    if len(chunks) > max_lines:
        lines.append(f"  … +{len(chunks) - max_lines} more")
    return "\n".join(lines)
