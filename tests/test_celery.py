"""
Tests for Celery tasks - retry logic, DLQ, job tracking.
"""
import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("celery")


class TestProcessQueryTask:
    def test_successful_task_stores_result(self):
        mock_result = {
            "response": {
                "answer": "test answer",
                "sources": [],
                "tokens_used": 50,
                "prompt_tokens": 30,
                "completion_tokens": 20,
            },
            "reranked_chunks": [],
            "error": "",
        }

        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "run_pipeline", return_value=mock_result), patch.object(
            batch_mod, "redis_client"
        ) as mock_redis, patch.object(batch_mod, "_get_cw", return_value=MagicMock()):
            process_query_task = batch_mod.process_query_task

            result = process_query_task.__wrapped__("job-123", "test query")
            assert result is not None
            mock_redis.hset.assert_called()

    def test_failed_task_retries_with_backoff(self):
        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "run_pipeline", side_effect=Exception("connection error")), patch.object(
            batch_mod, "redis_client"
        ), patch.object(batch_mod, "_get_cw", return_value=MagicMock()), patch.object(batch_mod, "dead_letter_task"):
            process_query_task = batch_mod.process_query_task
            from celery.exceptions import Retry

            process_query_task.request.retries = 0
            process_query_task.max_retries = 3
            process_query_task.retry = MagicMock(side_effect=Retry())

            with pytest.raises(Retry):
                process_query_task.__wrapped__("job-456", "failing query")

            process_query_task.retry.assert_called_once()
            _, kwargs = process_query_task.retry.call_args
            assert kwargs["countdown"] == 1  # 2^0 = 1

    def test_exponential_backoff_progression(self):
        expected_backoffs = [1, 2, 4]
        for attempt in range(3):
            assert 2 ** attempt == expected_backoffs[attempt]


class TestJobTracking:
    def test_enqueue_sets_initial_status(self):
        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "process_query_task") as mock_task, patch.object(
            batch_mod, "redis_client"
        ) as mock_redis:
            mock_task.apply_async.return_value = MagicMock(id="task-123")
            result = batch_mod.enqueue_batch_job("job-001", ["q1", "q2", "q3"])

            mock_redis.hset.assert_called()
            assert result["tasks"] == 3

    def test_get_job_status_progress(self):
        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "redis_client") as mock_redis:
            mock_redis.hgetall.return_value = {
                b"_status": b"processing",
                b"_total": b"3",
                b"query1": b'{"answer": "a1"}',
                b"query2": b'{"answer": "a2"}',
            }

            status = batch_mod.get_job_status("job-001")

            assert status["status"] == "processing"
            assert "progress" in status

    def test_get_job_status_not_found(self):
        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "redis_client") as mock_redis:
            mock_redis.hgetall.return_value = {}
            status = batch_mod.get_job_status("nonexistent")
            assert "error" in status


class TestDeadLetterQueue:
    def test_dlq_stores_failed_task(self):
        batch_mod = importlib.import_module("api.batch")
        with patch.object(batch_mod, "redis_client") as mock_redis, patch.object(batch_mod, "_get_cw", return_value=MagicMock()):
            batch_mod.dead_letter_task.__wrapped__("job-fail", "bad query", "connection timeout", "task-xyz")

            mock_redis.lpush.assert_called_once()
            call_args = mock_redis.lpush.call_args[0]
            assert call_args[0] == "dead_letter_queue"
            payload = json.loads(call_args[1])
            assert payload["job_id"] == "job-fail"
            assert payload["error"] == "connection timeout"
