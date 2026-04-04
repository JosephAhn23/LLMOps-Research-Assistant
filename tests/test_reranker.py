from agents.reranker import RerankerAgent


def test_reranker_fallback_preserves_input_order() -> None:
    """When the cross-encoder model is unavailable, rerank() returns the first
    top_k items in their original input order (no semantic scoring).
    This test verifies graceful degradation, NOT relevance ranking.
    """
    agent = RerankerAgent.__new__(RerankerAgent)
    agent._ready = False
    agent.cross_encoder = None
    agent.top_k = 1

    results = [
        {"text": "redis queue worker", "source": "a.md"},
        {"text": "orchestration with agents", "source": "b.md"},
    ]
    ranked = agent.rerank("agents orchestration", results)
    assert len(ranked) == 1
    # Passthrough returns the first item in input order — NOT the most relevant one.
    # "b.md" would win if the model were active; "a.md" wins here only because it
    # is first in the list.
    assert ranked[0]["source"] == "a.md"


def test_reranker_fallback_returns_all_when_top_k_exceeds_candidates() -> None:
    """Fallback should not crash when top_k > len(candidates)."""
    agent = RerankerAgent.__new__(RerankerAgent)
    agent._ready = False
    agent.cross_encoder = None
    agent.top_k = 10

    results = [{"text": "only one doc", "source": "x.md"}]
    ranked = agent.rerank("query", results)
    assert len(ranked) == 1


def test_reranker_does_not_mutate_input_candidates() -> None:
    """rerank() must not add rerank_score to the original candidate dicts."""
    agent = RerankerAgent.__new__(RerankerAgent)
    agent._ready = False
    agent.cross_encoder = None
    agent.top_k = 2

    original = [
        {"text": "doc a", "source": "a.md"},
        {"text": "doc b", "source": "b.md"},
    ]
    import copy
    snapshot = copy.deepcopy(original)
    agent.rerank("query", original)
    assert original == snapshot, "rerank() must not mutate the input list"
