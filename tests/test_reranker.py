from agents.reranker import RerankerAgent


def test_reranker_degrades_without_model() -> None:
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
    assert ranked[0]["source"] == "a.md"
