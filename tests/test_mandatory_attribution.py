"""Mandatory attribution IDs and grounding confidence."""

from context_engineering.mandatory_attribution import (
    build_attribution_footer,
    compute_grounding_confidence,
    enrich_chunks_with_attribution_ids,
)


def test_enrich_stable_ids() -> None:
    chunks = [{"text": "hello world", "source": "a.txt"}]
    out = enrich_chunks_with_attribution_ids(chunks)
    assert out[0]["attribution_id"].startswith("SRC-")
    assert out[0]["source_index"] == 1
    out2 = enrich_chunks_with_attribution_ids(chunks)
    assert out[0]["attribution_id"] == out2[0]["attribution_id"]


def test_grounding_confidence_range() -> None:
    chunks = [{"text": "The model uses attention mechanisms.", "source": "p.md"}]
    ans = "The model uses attention mechanisms [source_1]."
    g = compute_grounding_confidence(ans, chunks)
    assert 0.0 <= g <= 1.0


def test_footer_includes_ids() -> None:
    chunks = enrich_chunks_with_attribution_ids(
        [{"text": "x", "source": "s"}],
    )
    foot = build_attribution_footer(chunks)
    assert "SRC-" in foot
    assert "s" in foot
