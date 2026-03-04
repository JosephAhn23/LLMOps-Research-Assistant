"""
Prometheus metrics instrumentation for the LLMOps FastAPI gateway.

Exposes:
  - HTTP request latency histogram (by endpoint, method, status)
  - Token throughput counters (prompt tokens, completion tokens)
  - Inference latency histogram (by model, backend)
  - Queue depth gauge (Celery pending tasks)
  - Retrieval latency histogram (by backend: faiss, bm25, hybrid)
  - Active requests gauge
  - Error rate counter (by error type)
  - RAGAS score gauges (faithfulness, relevancy, precision, recall)

Usage — add to api/main.py:
    from observability.metrics import instrument_app, metrics_router
    instrument_app(app)
    app.include_router(metrics_router)

Then scrape http://localhost:8000/metrics with Prometheus.
"""
from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from typing import Callable

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ──────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed — metrics disabled. pip install prometheus-client")


def _make_metrics():
    """Create and register all Prometheus metrics."""
    if not _PROMETHEUS_AVAILABLE:
        return None

    class Metrics:
        # HTTP layer
        http_requests_total = Counter(
            "llmops_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        http_request_duration_seconds = Histogram(
            "llmops_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        http_active_requests = Gauge(
            "llmops_http_active_requests",
            "Number of in-flight HTTP requests",
        )

        # Inference
        inference_duration_seconds = Histogram(
            "llmops_inference_duration_seconds",
            "End-to-end inference latency",
            ["model", "backend"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        tokens_prompt_total = Counter(
            "llmops_tokens_prompt_total",
            "Total prompt tokens processed",
            ["model"],
        )
        tokens_completion_total = Counter(
            "llmops_tokens_completion_total",
            "Total completion tokens generated",
            ["model"],
        )
        tokens_per_second = Gauge(
            "llmops_tokens_per_second",
            "Current token generation throughput",
            ["model"],
        )

        # Retrieval
        retrieval_duration_seconds = Histogram(
            "llmops_retrieval_duration_seconds",
            "Retrieval latency",
            ["backend"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )
        retrieval_results_count = Histogram(
            "llmops_retrieval_results_count",
            "Number of documents retrieved",
            ["backend"],
            buckets=[1, 3, 5, 10, 20, 50],
        )

        # Queue
        queue_depth = Gauge(
            "llmops_queue_depth",
            "Celery task queue depth",
            ["queue_name"],
        )
        queue_tasks_total = Counter(
            "llmops_queue_tasks_total",
            "Total Celery tasks processed",
            ["queue_name", "status"],
        )

        # Errors
        errors_total = Counter(
            "llmops_errors_total",
            "Total errors by type",
            ["error_type", "endpoint"],
        )

        # Quality (RAGAS — updated by mlops/ragas_tracker.py)
        ragas_faithfulness = Gauge(
            "llmops_ragas_faithfulness",
            "Latest RAGAS faithfulness score",
        )
        ragas_answer_relevancy = Gauge(
            "llmops_ragas_answer_relevancy",
            "Latest RAGAS answer relevancy score",
        )
        ragas_context_precision = Gauge(
            "llmops_ragas_context_precision",
            "Latest RAGAS context precision score",
        )
        ragas_context_recall = Gauge(
            "llmops_ragas_context_recall",
            "Latest RAGAS context recall score",
        )

        # Model info
        model_info = Gauge(
            "llmops_model_info",
            "Model metadata (always 1, use labels for info)",
            ["model_name", "backend", "quantization"],
        )

    return Metrics()


# Singleton — import this from anywhere in the app
metrics = _make_metrics()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI middleware
# ──────────────────────────────────────────────────────────────────────────────

def instrument_app(app) -> None:
    """
    Add Prometheus instrumentation middleware to a FastAPI app.
    Call once at startup: instrument_app(app)
    """
    if not _PROMETHEUS_AVAILABLE or metrics is None:
        logger.warning("Prometheus not available — skipping instrumentation")
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            endpoint = request.url.path
            method = request.method

            metrics.http_active_requests.inc()
            start = time.perf_counter()
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            except Exception as exc:
                metrics.errors_total.labels(
                    error_type=type(exc).__name__, endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.perf_counter() - start
                metrics.http_active_requests.dec()
                metrics.http_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=str(status_code)
                ).inc()
                metrics.http_request_duration_seconds.labels(
                    method=method, endpoint=endpoint
                ).observe(duration)

    app.add_middleware(PrometheusMiddleware)
    logger.info("Prometheus middleware attached")


# ──────────────────────────────────────────────────────────────────────────────
# /metrics endpoint
# ──────────────────────────────────────────────────────────────────────────────

try:
    from fastapi import APIRouter
    from fastapi.responses import Response as FastAPIResponse

    metrics_router = APIRouter(tags=["observability"])

    @metrics_router.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus scrape endpoint."""
        if not _PROMETHEUS_AVAILABLE:
            return FastAPIResponse(content="# prometheus-client not installed\n", media_type="text/plain")
        return FastAPIResponse(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    @metrics_router.get("/health/metrics", tags=["observability"])
    async def metrics_health():
        """Quick check that metrics collection is active."""
        return {
            "prometheus_available": _PROMETHEUS_AVAILABLE,
            "metrics_endpoint": "/metrics",
        }

except ImportError:
    metrics_router = None


# ──────────────────────────────────────────────────────────────────────────────
# Context managers for timing blocks
# ──────────────────────────────────────────────────────────────────────────────

class track_inference:
    """
    Context manager that records inference latency and token counts.

    Usage:
        with track_inference(model="gpt-4o-mini", backend="openai") as t:
            result = llm.generate(prompt)
            t.prompt_tokens = len(prompt.split())
            t.completion_tokens = len(result.split())
    """
    def __init__(self, model: str = "unknown", backend: str = "unknown"):
        self.model = model
        self.backend = backend
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if metrics is None:
            return
        duration = time.perf_counter() - self._start
        metrics.inference_duration_seconds.labels(
            model=self.model, backend=self.backend
        ).observe(duration)
        if self.prompt_tokens:
            metrics.tokens_prompt_total.labels(model=self.model).inc(self.prompt_tokens)
        if self.completion_tokens:
            metrics.tokens_completion_total.labels(model=self.model).inc(self.completion_tokens)
            if duration > 0:
                metrics.tokens_per_second.labels(model=self.model).set(
                    self.completion_tokens / duration
                )


class track_retrieval:
    """
    Context manager that records retrieval latency and result count.

    Usage:
        with track_retrieval(backend="hybrid") as t:
            docs = retriever.search(query, top_k=10)
            t.result_count = len(docs)
    """
    def __init__(self, backend: str = "faiss"):
        self.backend = backend
        self.result_count: int = 0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if metrics is None:
            return
        duration = time.perf_counter() - self._start
        metrics.retrieval_duration_seconds.labels(backend=self.backend).observe(duration)
        if self.result_count:
            metrics.retrieval_results_count.labels(backend=self.backend).observe(
                self.result_count
            )


def update_ragas_scores(scores: dict) -> None:
    """
    Push latest RAGAS scores into Prometheus gauges.
    Call from mlops/ragas_tracker.py after each eval run.

    scores = {"faithfulness": 0.82, "answer_relevancy": 0.79, ...}
    """
    if metrics is None:
        return
    mapping = {
        "faithfulness": metrics.ragas_faithfulness,
        "answer_relevancy": metrics.ragas_answer_relevancy,
        "context_precision": metrics.ragas_context_precision,
        "context_recall": metrics.ragas_context_recall,
    }
    for key, gauge in mapping.items():
        if key in scores:
            gauge.set(scores[key])
    logger.debug("RAGAS scores pushed to Prometheus: %s", scores)


def update_queue_depth(queue_name: str, depth: int) -> None:
    """Update Celery queue depth gauge. Call from a periodic task."""
    if metrics is None:
        return
    metrics.queue_depth.labels(queue_name=queue_name).set(depth)
