"""
Traceable RAG — paragraph-level provenance and “model intuition” labels.

Maps each chunk to ``doc_id`` / ``ingested_at`` for audit trails (ingestion /
Delta-style pipelines can populate these on chunk metadata). Paragraphs in the
final report get inline provenance tags; claims that cannot be tied to retrieved
text are labeled **Model intuition** rather than **Grounded fact**.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

_PARA_SPLIT = re.compile(r"\n{2,}")


def normalize_chunk_provenance(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure ``doc_id`` and ``ingested_at`` exist for traceability."""
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    doc_id = chunk.get("doc_id") or chunk.get("id") or meta.get("doc_id") or chunk.get("source", "unknown")
    ingested = (
        chunk.get("ingested_at")
        or chunk.get("timestamp")
        or meta.get("ingested_at")
        or meta.get("timestamp")
        or ""
    )
    if not ingested:
        ingested = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {**chunk, "doc_id": str(doc_id), "ingested_at": str(ingested)}
    return out


def enrich_chunks_provenance(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_chunk_provenance(dict(c)) for c in chunks]


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)}


def _best_chunk_for_paragraph(para: str, chunks: Sequence[Dict[str, Any]]) -> Tuple[int, float]:
    pt = _tokens(para)
    if len(pt) < 2:
        return -1, 0.0
    best_i = -1
    best_score = 0.0
    for i, c in enumerate(chunks):
        ct = _tokens(c.get("text") or "")
        if not ct:
            continue
        j = len(pt & ct) / max(len(pt | ct), 1)
        if j > best_score:
            best_score = j
            best_i = i
    return best_i, best_score


def format_chunks_for_prompt(chunks: Sequence[Dict[str, Any]]) -> str:
    """RAG context lines including doc_id and timestamp for the model."""
    parts: List[str] = []
    for i, c in enumerate(chunks):
        did = c.get("doc_id", "?")
        ts = c.get("ingested_at", "?")
        src = c.get("source", f"doc_{i + 1}")
        parts.append(
            f"[source_{i + 1}] doc_id={did} ingested_at={ts} (from {src}):\n{c.get('text', '')}"
        )
    return "\n\n".join(parts)


def append_paragraph_provenance(
    answer: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    overlap_threshold: float = 0.06,
) -> str:
    """
    After each non-empty paragraph, append a bracketed provenance line.
    Weak overlap → ``[Model intuition — not mapped to bronze / shard record]``.
    """
    if not (answer or "").strip():
        return answer
    paras = [p.strip() for p in _PARA_SPLIT.split(answer.strip()) if p.strip()]
    if len(paras) <= 1 and "\n\n" not in answer:
        paras = [answer.strip()]

    out_parts: List[str] = []
    for para in paras:
        idx, score = _best_chunk_for_paragraph(para, chunks)
        if idx < 0 or score < overlap_threshold:
            tag = "[Model intuition — not mapped to bronze / shard record]"
        else:
            c = chunks[idx]
            did = c.get("doc_id", "?")
            ts = c.get("ingested_at", "?")
            tag = f"[Grounded fact — doc_id={did} ingested_at={ts} overlap={score:.2f}]"
        out_parts.append(f"{para}\n{tag}")
    return "\n\n".join(out_parts)


# RAGAS faithfulness gate (operational proxy unless batch metric is injected)
FAITHFULNESS_ALERT_THRESHOLD = float(os.getenv("FAITHFULNESS_ALERT_THRESHOLD", "0.8"))


def faithfulness_or_proxy(
    *,
    ragas_faithfulness: float | None = None,
    grounding_confidence: float | None = None,
) -> float:
    """
    Prefer an explicit RAGAS faithfulness score (e.g. from offline eval or MLflow);
    otherwise use lexical grounding confidence as a conservative proxy.
    """
    env_override = os.getenv("RAGAS_FAITHFULNESS_SCORE", "").strip()
    if env_override:
        try:
            return float(env_override)
        except ValueError:
            pass
    if ragas_faithfulness is not None:
        return float(ragas_faithfulness)
    if grounding_confidence is not None:
        return float(grounding_confidence)
    return 1.0


def low_confidence_human_review_message(effective_faithfulness: float) -> str | None:
    if effective_faithfulness < FAITHFULNESS_ALERT_THRESHOLD:
        return "Low Confidence: Needs Human Review"
    return None
