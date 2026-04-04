"""
Tests for FastAPI endpoints - realtime, batch, health, retrieve, ingest, auth.
"""
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.main.run_pipeline") as mock_run, \
         patch("api.batch.redis_client") as mock_redis, \
         patch("api.batch._get_cw", return_value=MagicMock()):
        mock_run.return_value = {"response": {}, "error": "", "reranked_chunks": []}
        mock_redis.ping.return_value = True
        mock_redis.hgetall.return_value = {}

        from api.main import app
        yield TestClient(app)


@pytest.fixture
def authed_client(monkeypatch):
    """Client fixture with API_KEY set; all requests must include the key."""
    monkeypatch.setenv("API_KEY", "test-secret")
    # Force re-evaluation of the module-level _API_KEY constant.
    import api.main as main_mod
    importlib.reload(main_mod)
    # Patch run_pipeline on the *reloaded* module so authenticated /query
    # requests don't hit the real pipeline (no FAISS index / OpenAI key in CI).
    with patch.object(main_mod, "run_pipeline",
                      return_value={"response": {}, "error": "", "reranked_chunks": []}), \
         patch("api.batch.redis_client") as mock_redis, \
         patch("api.batch._get_cw", return_value=MagicMock()):
        mock_redis.ping.return_value = True
        mock_redis.hgetall.return_value = {}
        yield TestClient(main_mod.app)


class TestRealtimeEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded")
        assert "checks" in body

    def test_query_endpoint_success(self, client):
        mock_result = {
            "response": {
                "answer": "The answer is 42.",
                "sources": ["doc1.pdf"],
                "tokens_used": 120,
                "prompt_tokens": 90,
                "completion_tokens": 30,
            },
            "error": "",
            "reranked_chunks": [{"text": "ctx", "source": "doc1.pdf", "rerank_score": 0.9}],
        }
        with patch("api.main.run_pipeline", return_value=mock_result):
            response = client.post("/query", json={"query": "what is the answer?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer is 42."
        assert "sources" in data

    def test_query_endpoint_error_returns_502(self, client):
        with patch("api.main.run_pipeline", return_value={"error": "index not loaded", "response": {}}):
            response = client.post("/query", json={"query": "test query"})
        assert response.status_code == 502

    def test_query_requires_query_field(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_endpoint_returns_job_id(self, client):
        # Patch where the function is *used* (api.main imports it at module level),
        # not where it is defined, so the mock is actually applied.
        with patch("api.main.enqueue_batch_job") as mock_enqueue:
            response = client.post("/batch", json={"queries": ["q1", "q2"]})
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        mock_enqueue.assert_called_once()
        _job_id, queries = mock_enqueue.call_args[0]
        assert queries == ["q1", "q2"]

    def test_batch_status_not_found_returns_404(self, client):
        with patch("api.batch.get_job_status", return_value={"error": "job not found"}):
            response = client.get("/batch/nonexistent-job-id")
        assert response.status_code == 404


class TestRetrieveEndpoint:
    def test_retrieve_returns_results(self, client):
        mock_chunks = [{"text": "chunk text", "source": "doc.md", "retrieval_score": 0.9}]
        mock_pipeline = MagicMock()
        mock_pipeline.retriever.retrieve = MagicMock(return_value=mock_chunks)
        mock_pipeline.reranker.rerank = MagicMock(return_value=mock_chunks)
        with patch("agents.orchestrator.get_pipeline", return_value=mock_pipeline):
            response = client.post("/retrieve", json={"query": "test query", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 3

    def test_retrieve_without_rerank(self, client):
        mock_chunks = [{"text": "chunk", "source": "doc.md", "retrieval_score": 0.8}]
        mock_pipeline = MagicMock()
        mock_pipeline.retriever.retrieve = MagicMock(return_value=mock_chunks)
        with patch("agents.orchestrator.get_pipeline", return_value=mock_pipeline):
            response = client.post("/retrieve", json={"query": "test", "rerank": False})
        assert response.status_code == 200
        mock_pipeline.reranker.rerank.assert_not_called()

    def test_retrieve_requires_query_field(self, client):
        response = client.post("/retrieve", json={})
        assert response.status_code == 422


class TestIngestEndpoint:
    def test_ingest_returns_doc_id(self, client):
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_documents = MagicMock()
        with patch("api.main._get_ingestion_pipeline", return_value=mock_pipeline):
            response = client.post("/ingest", json={"content": "some document text"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingested"
        assert "doc_id" in data

    def test_ingest_uses_metadata_source(self, client):
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_documents = MagicMock()
        with patch("api.main._get_ingestion_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/ingest",
                json={"content": "text", "metadata": {"source": "my-doc.pdf"}},
            )
        assert response.status_code == 200
        assert response.json()["source"] == "my-doc.pdf"

    def test_ingest_requires_content_field(self, client):
        response = client.post("/ingest", json={})
        assert response.status_code == 422


class TestAuthentication:
    def test_query_without_key_returns_401_when_key_set(self, authed_client):
        response = authed_client.post("/query", json={"query": "test"})
        assert response.status_code == 401

    def test_query_with_wrong_key_returns_401(self, authed_client):
        response = authed_client.post(
            "/query", json={"query": "test"}, headers={"X-API-Key": "wrong"}
        )
        assert response.status_code == 401

    def test_batch_status_without_key_returns_401_when_key_set(self, authed_client):
        response = authed_client.get("/batch/some-job-id")
        assert response.status_code == 401

    def test_health_is_public(self, authed_client):
        """Health endpoint should not require auth."""
        response = authed_client.get("/health")
        assert response.status_code == 200
