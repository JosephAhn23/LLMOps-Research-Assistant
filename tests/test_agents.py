"""
Tests for retriever, reranker, synthesizer, orchestrator.
"""
from unittest.mock import MagicMock, patch

import pytest


# ─── Retriever Tests ──────────────────────────────────────────


class TestRetrieverAgent:
    def _make_retriever(self, mock_metadata):
        from agents.retriever import RetrieverAgent

        agent = RetrieverAgent.__new__(RetrieverAgent)
        agent.top_k = 3
        agent.use_microservice_distributed = False
        agent.use_distributed = False
        agent.distributed_index = None

        agent.index = MagicMock()
        agent.index.search.return_value = (
            [[0.95, 0.87, 0.80]],
            [[0, 1, 2]],
        )
        agent.metadata = mock_metadata
        agent.embedder = MagicMock()
        agent.embedder.embed.return_value = [[0.1] * 384]
        return agent

    def test_retriever_returns_top_k(self):
        metadata = [{"text": f"chunk {i}", "source": f"doc_{i}", "chunk_id": str(i)} for i in range(5)]
        agent = self._make_retriever(metadata)
        results = agent.retrieve("test query")
        assert len(results) <= 3

    def test_retriever_includes_scores(self):
        metadata = [{"text": f"chunk {i}", "source": f"doc_{i}", "chunk_id": str(i)} for i in range(5)]
        agent = self._make_retriever(metadata)
        results = agent.retrieve("test query")
        for r in results:
            assert "retrieval_score" in r
            assert 0 <= r["retrieval_score"] <= 1.0

    def test_retriever_scores_descending(self):
        metadata = [{"text": f"chunk {i}", "source": f"doc_{i}", "chunk_id": str(i)} for i in range(5)]
        agent = self._make_retriever(metadata)
        results = agent.retrieve("test query")
        scores = [r["retrieval_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ─── Reranker Tests ───────────────────────────────────────────


class TestRerankerAgent:
    @pytest.fixture
    def mock_reranker(self):
        from agents.reranker import RerankerAgent

        agent = RerankerAgent.__new__(RerankerAgent)
        agent.top_k = 3
        agent._ready = True
        agent.cross_encoder = MagicMock()
        agent.cross_encoder.score_pairs.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]
        yield agent

    def test_reranker_returns_top_k(self, mock_reranker):
        candidates = [{"text": f"doc {i}", "source": f"s{i}", "retrieval_score": 0.5} for i in range(5)]
        results = mock_reranker.rerank("query", candidates)
        assert len(results) == 3

    def test_reranker_sorts_by_score(self, mock_reranker):
        candidates = [{"text": f"doc {i}", "source": f"s{i}", "retrieval_score": 0.5} for i in range(5)]
        results = mock_reranker.rerank("query", candidates)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reranker_adds_score_field(self, mock_reranker):
        candidates = [{"text": "doc", "source": "s0", "retrieval_score": 0.5}]
        results = mock_reranker.rerank("query", candidates)
        assert "rerank_score" in results[0]


# ─── Synthesizer Tests ────────────────────────────────────────


class TestSynthesizerAgent:
    def test_synthesizer_returns_answer(self):
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.model = "gpt-4o-mini"
        agent.max_tokens = 1024
        agent.backend = "openai"
        agent._ready = True
        agent.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is the answer [source_1]."
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        agent.client.chat.completions.create.return_value = mock_response

        chunks = [{"text": "relevant context", "source": "doc1"}]
        result = agent.synthesize("test query", chunks)
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_synthesizer_returns_sources(self):
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.backend = "openai"
        agent._ready = True
        agent.max_tokens = 1024
        agent.model = "gpt-4o-mini"
        agent.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Answer"
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        agent.client.chat.completions.create.return_value = mock_response

        chunks = [{"text": "ctx", "source": "doc1"}, {"text": "ctx2", "source": "doc2"}]
        result = agent.synthesize("query", chunks)
        assert result["sources"] == ["doc1", "doc2"]

    def test_synthesizer_empty_context(self):
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.backend = "openai"
        agent._ready = True
        agent.max_tokens = 1024
        agent.model = "gpt-4o-mini"
        agent.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "No context"
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        agent.client.chat.completions.create.return_value = mock_response

        result = agent.synthesize("query", [])
        assert "answer" in result


# ─── Orchestrator Tests ───────────────────────────────────────


class TestOrchestrator:
    def test_pipeline_state_flow(self):
        from agents.orchestrator import Pipeline

        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = [{"text": "ctx", "source": "s1", "retrieval_score": 0.9}]
        mock_rer = MagicMock()
        mock_rer.rerank.return_value = [{"text": "ctx", "source": "s1", "rerank_score": 0.95}]
        mock_syn = MagicMock()
        mock_syn.synthesize.return_value = {
            "answer": "answer",
            "sources": ["s1"],
            "tokens_used": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
        }

        pipe = Pipeline(retriever=mock_ret, reranker=mock_rer, synthesizer=mock_syn)
        result = pipe.run("test query")

        assert result.get("error") == "" or not result.get("error")
        assert "response" in result
        assert result["response"]["answer"] == "answer"

    def test_pipeline_handles_retriever_error(self):
        from agents.orchestrator import Pipeline

        bad_retriever = MagicMock()
        bad_retriever.retrieve.side_effect = Exception("FAISS index not loaded")

        pipe = Pipeline(
            retriever=bad_retriever,
            reranker=MagicMock(),
            synthesizer=MagicMock(),
        )
        result = pipe.run("test query")
        assert result.get("error") != ""
