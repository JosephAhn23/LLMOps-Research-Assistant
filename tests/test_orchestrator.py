from unittest.mock import MagicMock, patch

from agents.orchestrator import Pipeline, should_continue


# ─── Existing tests ────────────────────────────────────────────

def test_should_continue_routes_on_error() -> None:
    state = {"error": "failed"}
    assert should_continue(state) == "__end__"


def test_should_continue_routes_to_rerank() -> None:
    state = {"error": ""}
    assert should_continue(state) == "rerank"


# ─── Pipeline.run() ────────────────────────────────────────────

def _make_pipeline(
    retrieve_return=None,
    retrieve_side_effect=None,
    rerank_return=None,
    synthesize_return=None,
    synthesize_side_effect=None,
) -> Pipeline:
    retriever = MagicMock()
    if retrieve_side_effect:
        retriever.retrieve.side_effect = retrieve_side_effect
    else:
        retriever.retrieve.return_value = retrieve_return or [
            {"text": "doc", "source": "s.md", "retrieval_score": 0.9}
        ]

    reranker = MagicMock()
    reranker.rerank.return_value = rerank_return or [
        {"text": "doc", "source": "s.md", "rerank_score": 0.95}
    ]

    synthesizer = MagicMock()
    if synthesize_side_effect:
        synthesizer.synthesize.side_effect = synthesize_side_effect
    else:
        synthesizer.synthesize.return_value = synthesize_return or {
            "answer": "42",
            "sources": ["s.md"],
            "tokens_used": 10,
            "prompt_tokens": 8,
            "completion_tokens": 2,
        }

    return Pipeline(retriever, reranker, synthesizer)


def test_pipeline_run_success() -> None:
    pipeline = _make_pipeline()
    with patch("agents.orchestrator.mlflow"):
        result = pipeline.run("what is the answer?")
    assert result["response"]["answer"] == "42"
    assert result["error"] == ""


def test_pipeline_run_propagates_retrieval_error() -> None:
    """A retrieval failure must short-circuit the graph and surface the error."""
    pipeline = _make_pipeline(retrieve_side_effect=RuntimeError("index missing"))
    with patch("agents.orchestrator.mlflow"):
        result = pipeline.run("query")
    assert "index missing" in result["error"]
    pipeline.synthesizer.synthesize.assert_not_called()


def test_pipeline_run_propagates_synthesis_error() -> None:
    """A synthesis failure must surface the error without crashing."""
    pipeline = _make_pipeline(synthesize_side_effect=RuntimeError("LLM timeout"))
    with patch("agents.orchestrator.mlflow"):
        result = pipeline.run("query")
    assert "LLM timeout" in result["error"]


def test_pipeline_run_passes_query_to_retriever() -> None:
    pipeline = _make_pipeline()
    with patch("agents.orchestrator.mlflow"):
        pipeline.run("specific query text")
    pipeline.retriever.retrieve.assert_called_once_with("specific query text")


def test_pipeline_run_passes_retrieved_chunks_to_reranker() -> None:
    chunks = [{"text": "chunk1", "source": "a.md", "retrieval_score": 0.8}]
    pipeline = _make_pipeline(retrieve_return=chunks)
    with patch("agents.orchestrator.mlflow"):
        pipeline.run("query")
    pipeline.reranker.rerank.assert_called_once_with("query", chunks)


def test_pipeline_run_nested_mlflow_when_active_run_exists() -> None:
    """Pipeline.run() must not raise ActiveRunException when called inside
    an existing MLflow run (e.g. from a tracking decorator)."""
    pipeline = _make_pipeline()
    mock_run = MagicMock()
    with patch("agents.orchestrator.mlflow") as mock_mlflow:
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        pipeline.run("query")
    # nested=True must be passed when there is an active run
    mock_mlflow.start_run.assert_called_once_with(nested=True)
