"""
Production Celery - DLQ, Flower monitoring, retry backoff,
task routing, worker health checks.
Covers: Celery (real production depth)
"""
import json
import logging
import os
import time
import uuid
from typing import Dict

import redis
from celery import Celery
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded
from celery.signals import task_failure, task_success, worker_ready
from kombu import Exchange, Queue

from agents.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

_cw = None


def _get_cw():
    global _cw
    if _cw is None:
        from infra.aws_observability import CloudWatchObservability
        _cw = CloudWatchObservability()
    return _cw



# ─── Celery Config with task routing ──────────────────────────

celery_app = Celery("llmops", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    # Task routing - separate queues by priority
    task_queues=(
        Queue("high_priority", Exchange("high_priority"), routing_key="high"),
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("dead_letter", Exchange("dead_letter"), routing_key="dead"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Reliability settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timeouts
    task_soft_time_limit=30,
    task_time_limit=60,
    # Result retention
    result_expires=3600,
    # Worker concurrency
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


# ─── Signal Handlers ──────────────────────────────────────────

@task_success.connect
def on_task_success(sender, result, **kwargs):
    _get_cw().log_structured(
        "celery-tasks",
        {
            "event": "task_success",
            "task": sender.name,
            "result_size": len(str(result)),
        },
    )


@task_failure.connect
def on_task_failure(sender, task_id, exception, traceback, **kwargs):
    _get_cw().log_structured(
        "celery-tasks",
        {
            "event": "task_failure",
            "task": sender.name,
            "task_id": task_id,
            "error": str(exception),
        },
    )


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logger.info(f"Worker ready: {sender}")


# ─── Dead Letter Queue ────────────────────────────────────────

@celery_app.task(queue="dead_letter", name="tasks.dead_letter")
def dead_letter_task(job_id: str, query: str, error: str, original_task_id: str):
    """Permanently failed tasks land here for manual inspection."""
    payload = {
        "job_id": job_id,
        "query": query,
        "error": error,
        "original_task_id": original_task_id,
        "failed_at": time.time(),
    }
    redis_client.lpush("dead_letter_queue", json.dumps(payload))
    redis_client.expire("dead_letter_queue", 86400 * 7)

    _get_cw().log_structured(
        "celery-dlq",
        {
            "event": "dead_letter",
            "job_id": job_id,
            "error": error,
        },
    )
    logger.error(f"Task permanently failed - DLQ: {job_id} | {error}")


# ─── Main Task ────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.process_query",
    queue="default",
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=30,
    time_limit=60,
)
def process_query_task(self, job_id: str, query: str) -> Dict:
    start = time.time()
    try:
        result = run_pipeline(query)

        latency = (time.time() - start) * 1000
        from infra.aws_observability import InferenceMetrics
        _get_cw().emit_inference_metrics(
            InferenceMetrics(
                latency_ms=latency,
                tokens_generated=result.get("response", {}).get("tokens_used", 0),
                tokens_prompt=result.get("response", {}).get("prompt_tokens", 0),
                retrieval_latency_ms=0,
                rerank_latency_ms=0,
                n_chunks_retrieved=len(result.get("reranked_chunks", [])),
                cache_hit=False,
                model_backend="gpt-4o-mini",
                error=bool(result.get("error")),
            )
        )

        task_id = self.request.id or str(uuid.uuid4())
        redis_client.hset(
            f"job:{job_id}",
            mapping={
                f"result:{task_id}": json.dumps(result.get("response", {})),
                "_status": "completed",
                "_latency_ms": str(round(latency, 2)),
            },
        )
        redis_client.expire(f"job:{job_id}", 3600)
        return result

    except SoftTimeLimitExceeded:
        logger.warning(f"Soft timeout hit for job {job_id}")
        raise self.retry(countdown=5, max_retries=2)

    except Exception as exc:
        retry_count = self.request.retries
        backoff = 2 ** retry_count

        _get_cw().log_structured(
            "celery-errors",
            {
                "event": "task_retry",
                "job_id": job_id,
                "attempt": retry_count + 1,
                "error": str(exc),
                "next_retry_in": backoff,
            },
        )

        if retry_count >= self.max_retries:
            dead_letter_task.apply_async(
                args=[job_id, query, str(exc), self.request.id],
                queue="dead_letter",
            )
            redis_client.hset(f"job:{job_id}", "_status", "failed")
            raise MaxRetriesExceededError(f"Task {job_id} permanently failed")

        raise self.retry(exc=exc, countdown=backoff)


# ─── High Priority Task ───────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.process_query_priority",
    queue="high_priority",
    max_retries=5,
    soft_time_limit=15,
    time_limit=30,
)
def process_query_priority_task(self, job_id: str, query: str) -> Dict:
    """High priority queue - SLA-bound queries.

    Delegates to the same pipeline logic as the default task but runs in the
    high_priority queue with a tighter SLA (15 s soft / 30 s hard limit).
    We call run_pipeline directly rather than invoking process_query_task as a
    plain function, which would bypass Celery's retry/context machinery.
    """
    start = time.time()
    try:
        result = run_pipeline(query)

        latency = (time.time() - start) * 1000
        from infra.aws_observability import InferenceMetrics
        _get_cw().emit_inference_metrics(
            InferenceMetrics(
                latency_ms=latency,
                tokens_generated=result.get("response", {}).get("tokens_used", 0),
                tokens_prompt=result.get("response", {}).get("prompt_tokens", 0),
                retrieval_latency_ms=0,
                rerank_latency_ms=0,
                n_chunks_retrieved=len(result.get("reranked_chunks", [])),
                cache_hit=False,
                model_backend="gpt-4o-mini",
                error=bool(result.get("error")),
            )
        )

        task_id = self.request.id or str(uuid.uuid4())
        redis_client.hset(
            f"job:{job_id}",
            mapping={
                f"result:{task_id}": json.dumps(result.get("response", {})),
                "_status": "completed",
                "_latency_ms": str(round(latency, 2)),
            },
        )
        redis_client.expire(f"job:{job_id}", 3600)
        return result

    except SoftTimeLimitExceeded:
        logger.warning("Soft timeout hit for priority job %s", job_id)
        raise self.retry(countdown=2, max_retries=2)

    except Exception as exc:
        retry_count = self.request.retries
        backoff = 2 ** retry_count

        if retry_count >= self.max_retries:
            dead_letter_task.apply_async(
                args=[job_id, query, str(exc), self.request.id],
                queue="dead_letter",
            )
            redis_client.hset(f"job:{job_id}", "_status", "failed")
            raise MaxRetriesExceededError(f"Priority task {job_id} permanently failed")

        raise self.retry(exc=exc, countdown=backoff)


# ─── Job Management ───────────────────────────────────────────

def enqueue_batch_job(
    job_id: str,
    queries: list,
    priority: bool = False,
) -> Dict:
    """Enqueue batch with optional priority routing."""
    redis_client.hset(
        f"job:{job_id}",
        mapping={
            "_status": "queued",
            "_total": len(queries),
            "_queued_at": str(time.time()),
        },
    )

    task_fn = process_query_priority_task if priority else process_query_task

    task_ids = []
    for query in queries:
        result = task_fn.apply_async(args=[job_id, query])
        task_ids.append(result.id)

    redis_client.hset(f"job:{job_id}", "_task_ids", json.dumps(task_ids))
    return {"job_id": job_id, "tasks": len(queries), "priority": priority}


def get_job_status(job_id: str) -> Dict:
    data = redis_client.hgetall(f"job:{job_id}")
    if not data:
        return {"error": "job not found"}

    decoded = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in data.items()
    }

    completed = sum(1 for k in decoded if k.startswith("result:"))
    total = int(decoded.get("_total", 0))

    return {
        "job_id": job_id,
        "status": decoded.get("_status", "unknown"),
        "progress": f"{completed}/{total}",
        "completion_pct": round(completed / max(total, 1) * 100, 1),
        "queued_at": decoded.get("_queued_at"),
    }


def get_dead_letter_queue(limit: int = 100) -> list:
    """Inspect permanently failed tasks."""
    items = redis_client.lrange("dead_letter_queue", 0, limit - 1)
    return [json.loads(item) for item in items]


def retry_dead_letter(job_id: str, query: str):
    """Manually retry a dead-lettered task."""
    new_job_id = str(uuid.uuid4())
    process_query_task.apply_async(args=[new_job_id, query], queue="default")
    return {"new_job_id": new_job_id, "status": "requeued"}
