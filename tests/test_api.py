"""
Tests for FastAPI endpoints - realtime, batch, health.
"""
from unittest.mock import patch, MagicMock

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
        with patch("api.batch.enqueue_batch_job"):
            response = client.post("/batch", json={"queries": ["q1", "q2"]})
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_batch_status_not_found_returns_404(self, client):
        with patch("api.batch.get_job_status", return_value={"error": "job not found"}):
            response = client.get("/batch/nonexistent-job-id")
        assert response.status_code == 404
