"""Traceable RAG provenance helpers."""

from context_engineering.traceable_rag import (
    append_paragraph_provenance,
    enrich_chunks_provenance,
    faithfulness_or_proxy,
    format_chunks_for_prompt,
    low_confidence_human_review_message,
    normalize_chunk_provenance,
)


def test_normalize_chunk_provenance_doc_id() -> None:
    c = normalize_chunk_provenance({"text": "hi", "id": "doc-1", "source": "s"})
    assert c["doc_id"] == "doc-1"
    assert "ingested_at" in c


def test_format_chunks_for_prompt_includes_ids() -> None:
    chunks = enrich_chunks_provenance([{"text": "alpha", "id": "d1", "source": "f.md"}])
    s = format_chunks_for_prompt(chunks)
    assert "doc_id=d1" in s
    assert "ingested_at" in s


def test_append_paragraph_provenance_grounded_vs_intuition() -> None:
    chunks = enrich_chunks_provenance(
        [{"text": "The speed of light in vacuum is constant.", "id": "physics", "source": "p.md"}],
    )
    out = append_paragraph_provenance(
        "The speed of light in vacuum is constant.\n\nRandom unrelated fiction about dragons.",
        chunks,
        overlap_threshold=0.04,
    )
    assert "Grounded fact" in out
    assert "Model intuition" in out


def test_faithfulness_proxy_and_alert() -> None:
    assert faithfulness_or_proxy(grounding_confidence=0.9) == 0.9
    assert low_confidence_human_review_message(0.5) == "Low Confidence: Needs Human Review"
    assert low_confidence_human_review_message(0.95) is None
