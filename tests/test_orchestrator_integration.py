from agents.orchestrator import Pipeline


class _StubRetriever:
    def retrieve(self, query: str) -> list[dict]:
        return [
            {"text": f"match for {query}", "source": "doc-a.md", "retrieval_score": 0.9},
            {"text": "other context", "source": "doc-b.md", "retrieval_score": 0.4},
        ]


class _StubReranker:
    def rerank(self, _query: str, candidates: list[dict]) -> list[dict]:
        return sorted(candidates, key=lambda x: x["retrieval_score"], reverse=True)[:1]


class _StubSynthesizer:
    def synthesize(self, query: str, chunks: list[dict]) -> dict:
        return {
            "answer": f"answer for {query}",
            "sources": [c["source"] for c in chunks],
            "tokens_used": 42,
            "prompt_tokens": 30,
            "completion_tokens": 12,
        }


class _FailingRetriever:
    def retrieve(self, _query: str) -> list[dict]:
        raise RuntimeError("retriever unavailable")


def test_pipeline_happy_path() -> None:
    pipe = Pipeline(
        retriever=_StubRetriever(),
        reranker=_StubReranker(),
        synthesizer=_StubSynthesizer(),
    )
    result = pipe.run("what is the architecture")

    assert result["error"] == ""
    assert result["response"]["answer"].startswith("answer for")
    assert result["response"]["sources"] == ["doc-a.md"]
    assert len(result["reranked_chunks"]) == 1


def test_pipeline_short_circuits_on_retriever_failure() -> None:
    pipe = Pipeline(
        retriever=_FailingRetriever(),
        reranker=_StubReranker(),
        synthesizer=_StubSynthesizer(),
    )
    result = pipe.run("query")

    assert "retriever unavailable" in result["error"]
    assert result["reranked_chunks"] == []
    assert result["response"] == {}


def test_pipeline_protocols_are_satisfied() -> None:
    from agents.protocols import Reranker, Retriever, Synthesizer

    assert isinstance(_StubRetriever(), Retriever)
    assert isinstance(_StubReranker(), Reranker)
    assert isinstance(_StubSynthesizer(), Synthesizer)
