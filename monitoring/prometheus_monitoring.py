"""
Production observability stack for LLMOps Research Assistant.

Components:
  - Prometheus metrics  — request latency, token counts, queue depth,
                          retrieval scores, error rates, GPU memory, NDCG
  - FastAPI middleware  — auto-instruments every endpoint via make_asgi_app()
  - Grafana dashboard  — programmatically built JSON (import via UI or API)
  - Alert rules        — latency SLO, error budget, quality regression alerts
  - NVIDIA DCGM stub   — GPU metrics bridge (requires dcgm-exporter on GPU node)

Metrics exposed at /metrics (Prometheus scrape endpoint):

  llmops_request_latency_seconds   histogram  endpoint, model, status
  llmops_retrieval_latency_seconds histogram  method (bm25/faiss/hybrid)
  llmops_time_to_first_token_seconds histogram model
  llmops_token_count_total         counter    token_type (prompt/completion), model
  llmops_requests_total            counter    endpoint, model, status
  llmops_errors_total              counter    endpoint, error_type
  llmops_cache_hit_total           counter    cache_type
  llmops_dlq_events_total          counter    topic
  llmops_queue_depth               gauge      queue (realtime/batch)
  llmops_active_requests           gauge      endpoint
  llmops_gpu_memory_used_bytes     gauge      device
  llmops_model_loaded              gauge      model
  llmops_ragas_score               gauge      metric (faithfulness/relevancy/...)
  llmops_deepeval_score            gauge      metric
  llmops_retrieval_score           histogram  method
  llmops_ndcg                      gauge      k

docker-compose addition:
  prometheus:
    image: prom/prometheus
    volumes: [./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml]
  grafana:
    image: grafana/grafana
    volumes: [./monitoring/grafana:/etc/grafana/provisioning]

Usage:
    from monitoring.prometheus_monitoring import metrics, instrument_app

    instrument_app(app)   # FastAPI app — adds middleware + /metrics

    metrics.record_inference(latency_ms=142, prompt_tokens=50,
                             completion_tokens=200, model="gpt-4o-mini")
    metrics.set_ragas_scores({"faithfulness": 0.847, "answer_relevancy": 0.823})
    metrics.set_ndcg(k=10, value=0.82)

    with metrics.measure("generate", model="gpt-4o-mini"):
        answer = pipeline.run(query)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Metric registry
# ──────────────────────────────────────────────────────────────────────────────

class LLMOpsMetrics:
    """
    Central Prometheus metric registry for the LLMOps pipeline.

    All metrics are lazy-initialised so the module can be imported
    without prometheus_client installed (graceful degradation to no-ops).
    """

    def __init__(self):
        self._initialised = False
        self._metrics: Dict = {}

    def _init(self):
        if self._initialised:
            return
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY

            self._registry = REGISTRY

            # ── Latency histograms ─────────────────────────────────────────
            self._metrics["request_latency"] = Histogram(
                "llmops_request_latency_seconds",
                "End-to-end request latency",
                labelnames=["endpoint", "model", "status"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            )
            self._metrics["retrieval_latency"] = Histogram(
                "llmops_retrieval_latency_seconds",
                "Retrieval pipeline latency",
                labelnames=["method"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            )
            self._metrics["ttft"] = Histogram(
                "llmops_time_to_first_token_seconds",
                "Time to first token (streaming)",
                labelnames=["model"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            )
            self._metrics["retrieval_score"] = Histogram(
                "llmops_retrieval_score",
                "Retrieval relevance scores",
                labelnames=["method"],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )

            # ── Counters ───────────────────────────────────────────────────
            self._metrics["tokens"] = Counter(
                "llmops_token_count_total",
                "Total tokens processed",
                labelnames=["token_type", "model"],
            )
            self._metrics["requests"] = Counter(
                "llmops_requests_total",
                "Total requests",
                labelnames=["endpoint", "model", "status"],
            )
            self._metrics["errors"] = Counter(
                "llmops_errors_total",
                "Total errors",
                labelnames=["endpoint", "error_type"],
            )
            self._metrics["cache_hits"] = Counter(
                "llmops_cache_hit_total",
                "Cache hits",
                labelnames=["cache_type"],
            )
            self._metrics["dlq_events"] = Counter(
                "llmops_dlq_events_total",
                "Events sent to dead letter queue",
                labelnames=["topic"],
            )

            # ── Gauges ─────────────────────────────────────────────────────
            self._metrics["queue_depth"] = Gauge(
                "llmops_queue_depth",
                "Current queue depth",
                labelnames=["queue"],
            )
            self._metrics["active_requests"] = Gauge(
                "llmops_active_requests",
                "Currently active requests",
                labelnames=["endpoint"],
            )
            self._metrics["gpu_memory_used_bytes"] = Gauge(
                "llmops_gpu_memory_used_bytes",
                "GPU memory currently allocated",
                labelnames=["device"],
            )
            self._metrics["model_loaded"] = Gauge(
                "llmops_model_loaded",
                "Whether a model is currently loaded (1=yes, 0=no)",
                labelnames=["model"],
            )

            # ── Quality metrics ────────────────────────────────────────────
            self._metrics["ragas_score"] = Gauge(
                "llmops_ragas_score",
                "Latest RAGAS evaluation score",
                labelnames=["metric"],
            )
            self._metrics["deepeval_score"] = Gauge(
                "llmops_deepeval_score",
                "Latest DeepEval evaluation score",
                labelnames=["metric"],
            )
            self._metrics["ndcg"] = Gauge(
                "llmops_ndcg",
                "Latest NDCG@K score",
                labelnames=["k"],
            )

            self._initialised = True
            logger.info("Prometheus metrics initialised.")

        except ImportError:
            logger.warning(
                "prometheus_client not installed — metrics disabled. "
                "pip install prometheus-client"
            )
            self._initialised = True  # avoid repeated attempts

    def _get(self, name: str):
        self._init()
        return self._metrics.get(name)

    # ── Recording API ────────────────────────────────────────────────────────

    def record_request(
        self,
        endpoint: str,
        model: str,
        latency_ms: float,
        status: str = "success",
    ) -> None:
        m = self._get("request_latency")
        if m:
            m.labels(endpoint=endpoint, model=model, status=status).observe(latency_ms / 1000)
        c = self._get("requests")
        if c:
            c.labels(endpoint=endpoint, model=model, status=status).inc()

    def record_inference(
        self,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        endpoint: str = "generate",
        ttft_ms: Optional[float] = None,
    ) -> None:
        """Record a complete inference call — latency, tokens, and TTFT."""
        self.record_request(endpoint, model, latency_ms)
        t = self._get("tokens")
        if t:
            t.labels(token_type="prompt", model=model).inc(prompt_tokens)
            t.labels(token_type="completion", model=model).inc(completion_tokens)
        if ttft_ms is not None:
            h = self._get("ttft")
            if h:
                h.labels(model=model).observe(ttft_ms / 1000)

    def record_retrieval(
        self,
        method: str,
        latency_ms: float,
        scores: Optional[List[float]] = None,
    ) -> None:
        """Record retrieval latency and per-document relevance scores."""
        h = self._get("retrieval_latency")
        if h:
            h.labels(method=method).observe(latency_ms / 1000)
        if scores:
            s = self._get("retrieval_score")
            if s:
                for score in scores:
                    s.labels(method=method).observe(score)

    def record_error(self, endpoint: str, error_type: str) -> None:
        c = self._get("errors")
        if c:
            c.labels(endpoint=endpoint, error_type=error_type).inc()

    def record_cache_hit(self, cache_type: str = "semantic") -> None:
        c = self._get("cache_hits")
        if c:
            c.labels(cache_type=cache_type).inc()

    def record_dlq_event(self, topic: str) -> None:
        c = self._get("dlq_events")
        if c:
            c.labels(topic=topic).inc()

    def set_queue_depth(self, queue: str, depth: int) -> None:
        g = self._get("queue_depth")
        if g:
            g.labels(queue=queue).set(depth)

    def set_model_loaded(self, model: str, loaded: bool = True) -> None:
        g = self._get("model_loaded")
        if g:
            g.labels(model=model).set(1 if loaded else 0)

    def set_ragas_scores(self, scores: Dict[str, float]) -> None:
        """Push latest RAGAS scores into Prometheus gauges."""
        g = self._get("ragas_score")
        if g:
            for metric, value in scores.items():
                g.labels(metric=metric).set(value)

    def set_deepeval_scores(self, scores: Dict[str, float]) -> None:
        g = self._get("deepeval_score")
        if g:
            for metric, value in scores.items():
                g.labels(metric=metric).set(value)

    def set_ndcg(self, k: int, value: float) -> None:
        g = self._get("ndcg")
        if g:
            g.labels(k=str(k)).set(value)

    def update_gpu_memory(self) -> None:
        """Snapshot current GPU memory allocation for all visible devices."""
        g = self._get("gpu_memory_used_bytes")
        if not g:
            return
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                used = torch.cuda.memory_allocated(i)
                g.labels(device=f"cuda:{i}").set(used)
        except Exception:
            pass

    # ── Context managers & decorators ────────────────────────────────────────

    @contextmanager
    def measure(self, endpoint: str, model: str = "unknown"):
        """
        Context manager for timing code blocks.

        with metrics.measure("generate", model="gpt-4o-mini"):
            answer = pipeline.run(query)
        """
        t0 = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception as e:
            status = "error"
            self.record_error(endpoint, type(e).__name__)
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.record_request(endpoint, model, latency_ms, status)

    def track_active(self, endpoint: str):
        """
        Async decorator that tracks in-flight request count.

        @metrics.track_active("query")
        async def query_endpoint(req: QueryRequest):
            ...
        """
        def decorator(fn):
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                g = self._get("active_requests")
                if g:
                    g.labels(endpoint=endpoint).inc()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    if g:
                        g.labels(endpoint=endpoint).dec()
            return wrapper
        return decorator


# Singleton — import from anywhere in the codebase
metrics = LLMOpsMetrics()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI instrumentation
# ──────────────────────────────────────────────────────────────────────────────

def instrument_app(app):
    """
    Add Prometheus middleware and /metrics endpoint to a FastAPI app.
    Uses prometheus_client.make_asgi_app() — the recommended approach.

    Call once at startup:
        from monitoring.prometheus_monitoring import metrics, instrument_app
        instrument_app(app)
    """
    try:
        from prometheus_client import make_asgi_app
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        class _PrometheusMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                if request.url.path == "/metrics":
                    return await call_next(request)
                endpoint = request.url.path
                model = request.headers.get("X-Model", "unknown")
                t0 = time.perf_counter()
                status = "success"
                try:
                    response = await call_next(request)
                    status = str(response.status_code)
                    return response
                except Exception as e:
                    status = "error"
                    metrics.record_error(endpoint, type(e).__name__)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    metrics.record_request(endpoint, model, latency_ms, status)

        app.add_middleware(_PrometheusMiddleware)
        app.mount("/metrics", make_asgi_app())
        logger.info("Prometheus middleware + /metrics endpoint mounted.")

    except ImportError:
        logger.warning("prometheus_client not installed — skipping instrumentation.")

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Config file content
# ──────────────────────────────────────────────────────────────────────────────

PROMETHEUS_CONFIG = """\
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/alerts.yml

scrape_configs:
  - job_name: llmops-api
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics

  - job_name: llmops-celery
    static_configs:
      - targets: ['celery-exporter:9540']

  - job_name: dcgm-exporter
    static_configs:
      - targets: ['dcgm-exporter:9400']

  - job_name: node-exporter
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: redis-exporter
    static_configs:
      - targets: ['redis-exporter:9121']
"""

ALERT_RULES = """\
groups:
  - name: llmops-slo
    rules:

      - alert: HighLatencyP99
        expr: histogram_quantile(0.99, rate(llmops_request_latency_seconds_bucket[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency > 10s"
          description: "Endpoint {{ $labels.endpoint }} p99={{ $value | humanizeDuration }}"

      - alert: HighErrorRate
        expr: rate(llmops_errors_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5%"
          description: "Error rate on {{ $labels.endpoint }}: {{ $value | humanize }}/s"

      - alert: QualityRegression
        expr: llmops_ragas_score{metric="faithfulness"} < 0.70
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAGAS faithfulness below threshold (0.70)"
          description: "Current faithfulness: {{ $value | humanize }}"

      - alert: QueueDepthHigh
        expr: llmops_queue_depth > 100
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Queue depth > 100"
          description: "Queue {{ $labels.queue }}: {{ $value }} items"

      - alert: GPUMemoryHigh
        expr: llmops_gpu_memory_used_bytes / (1024 * 1024 * 1024) > 20
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory > 20 GB"
          description: "Device {{ $labels.device }}: {{ $value | humanize }}GB used"

      - alert: NDCGDrop
        expr: llmops_ndcg{k="10"} < 0.60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "NDCG@10 dropped below 0.60"
          description: "Current NDCG@10: {{ $value | humanize }}"
"""


# ──────────────────────────────────────────────────────────────────────────────
# Grafana dashboard builder
# ──────────────────────────────────────────────────────────────────────────────

def build_grafana_dashboard() -> Dict:
    """
    Build a complete Grafana dashboard JSON for LLMOps monitoring.
    Import via Grafana UI: Dashboards → Import → paste JSON.
    Or use the provisioning files generated by generate_monitoring_files().
    """

    def timeseries(title, expr, unit, panel_id, x, y, w=12, h=8, legend="{{endpoint}}"):
        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "fieldConfig": {
                "defaults": {"unit": unit, "color": {"mode": "palette-classic"}},
                "overrides": [],
            },
            "targets": [{"expr": expr, "legendFormat": legend, "refId": "A"}],
            "options": {"tooltip": {"mode": "multi"}},
        }

    def stat(title, expr, unit, panel_id, x, y, w=6, h=4, thresholds=None):
        th = thresholds or [{"color": "green", "value": None}]
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "color": {"mode": "thresholds"},
                    "thresholds": {"mode": "absolute", "steps": th},
                },
                "overrides": [],
            },
            "targets": [{"expr": expr, "legendFormat": "", "refId": "A"}],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        }

    def gauge(title, expr, unit, panel_id, x, y, min_val=0, max_val=1, w=6, h=6):
        return {
            "id": panel_id,
            "title": title,
            "type": "gauge",
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "fieldConfig": {
                "defaults": {
                    "unit": unit, "min": min_val, "max": max_val,
                    "color": {"mode": "thresholds"},
                    "thresholds": {"mode": "absolute", "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 0.65},
                        {"color": "green", "value": 0.75},
                    ]},
                },
                "overrides": [],
            },
            "targets": [{"expr": expr, "legendFormat": "{{metric}}", "refId": "A"}],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        }

    def row(title, panel_id, y):
        return {"id": panel_id, "title": title, "type": "row",
                "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "collapsed": False}

    panels = [
        row("HTTP Gateway", 100, 0),
        stat("Request Rate (QPS)", 'sum(rate(llmops_requests_total[1m]))', "reqps",
             1, 0, 1, 6, 4,
             [{"color": "green", "value": None}, {"color": "yellow", "value": 50}, {"color": "red", "value": 200}]),
        stat("Active Requests", 'sum(llmops_active_requests)', "short",
             2, 6, 1, 6, 4,
             [{"color": "green", "value": None}, {"color": "yellow", "value": 20}, {"color": "red", "value": 50}]),
        stat("Error Rate (/s)", 'sum(rate(llmops_errors_total[5m]))', "reqps",
             3, 12, 1, 6, 4,
             [{"color": "green", "value": None}, {"color": "red", "value": 0.05}]),
        stat("Cache Hit Rate", 'sum(rate(llmops_cache_hit_total[5m])) / sum(rate(llmops_requests_total[5m]))',
             "percentunit", 4, 18, 1, 6, 4),
        timeseries("P50 / P95 / P99 Latency", """histogram_quantile(0.50, sum(rate(llmops_request_latency_seconds_bucket[5m])) by (le))
or histogram_quantile(0.95, sum(rate(llmops_request_latency_seconds_bucket[5m])) by (le))
or histogram_quantile(0.99, sum(rate(llmops_request_latency_seconds_bucket[5m])) by (le))""",
             "s", 5, 0, 5, 24, 8, "p{{quantile}}"),

        row("Inference & Tokens", 101, 13),
        timeseries("Token Throughput (tok/s)", 'sum(rate(llmops_token_count_total[1m])) by (token_type, model)',
                   "short", 6, 0, 14, 12, 8, "{{token_type}} {{model}}"),
        timeseries("Time to First Token (p95)", 'histogram_quantile(0.95, sum(rate(llmops_time_to_first_token_seconds_bucket[5m])) by (le, model))',
                   "s", 7, 12, 14, 12, 8, "{{model}}"),

        row("Retrieval", 102, 22),
        timeseries("Retrieval Latency p95 by Method", 'histogram_quantile(0.95, sum(rate(llmops_retrieval_latency_seconds_bucket[5m])) by (le, method))',
                   "s", 8, 0, 23, 12, 8, "{{method}}"),
        timeseries("Retrieval Score Distribution (median)", 'histogram_quantile(0.50, sum(rate(llmops_retrieval_score_bucket[5m])) by (le, method))',
                   "percentunit", 9, 12, 23, 12, 8, "{{method}}"),

        row("RAG Quality (RAGAS / DeepEval / NDCG)", 103, 31),
        gauge("Faithfulness", 'llmops_ragas_score{metric="faithfulness"}', "percentunit", 10, 0, 32),
        gauge("Answer Relevancy", 'llmops_ragas_score{metric="answer_relevancy"}', "percentunit", 11, 6, 32),
        gauge("Context Precision", 'llmops_ragas_score{metric="context_precision"}', "percentunit", 12, 12, 32),
        gauge("Context Recall", 'llmops_ragas_score{metric="context_recall"}', "percentunit", 13, 18, 32),
        timeseries("RAGAS Score Trends", 'llmops_ragas_score', "percentunit", 14, 0, 38, 16, 8, "{{metric}}"),
        stat("NDCG@10", 'llmops_ndcg{k="10"}', "percentunit", 15, 16, 38, 8, 8,
             [{"color": "red", "value": None}, {"color": "yellow", "value": 0.6}, {"color": "green", "value": 0.75}]),

        row("Queue & GPU", 104, 46),
        timeseries("Queue Depth", 'llmops_queue_depth', "short", 16, 0, 47, 12, 8, "{{queue}}"),
        timeseries("GPU Memory Used (GB)", 'llmops_gpu_memory_used_bytes / (1024*1024*1024)', "decgbytes",
                   17, 12, 47, 12, 8, "{{device}}"),
        timeseries("DLQ Events (/s)", 'rate(llmops_dlq_events_total[5m])', "reqps", 18, 0, 55, 12, 8, "{{topic}}"),
    ]

    return {
        "title": "LLMOps Research Assistant",
        "uid": "llmops-main",
        "tags": ["llmops", "ml", "rag", "prometheus"],
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
        "schemaVersion": 38,
        "version": 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# File generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_monitoring_files(output_dir: str = "monitoring") -> List[str]:
    """
    Write all monitoring config files to disk.
    Safe to re-run — overwrites existing files.
    """
    base = Path(output_dir)

    # Prometheus
    (base / "prometheus.yml").write_text(PROMETHEUS_CONFIG)
    (base / "alerts.yml").write_text(ALERT_RULES)

    # Grafana provisioning
    (base / "grafana" / "dashboards").mkdir(parents=True, exist_ok=True)
    (base / "grafana" / "datasources").mkdir(parents=True, exist_ok=True)

    (base / "grafana" / "dashboards" / "llmops.json").write_text(
        json.dumps(build_grafana_dashboard(), indent=2)
    )
    (base / "grafana" / "datasources" / "prometheus.yml").write_text(
        "apiVersion: 1\n"
        "datasources:\n"
        "  - name: Prometheus\n"
        "    type: prometheus\n"
        "    url: http://prometheus:9090\n"
        "    isDefault: true\n"
        "    uid: prometheus\n"
    )
    (base / "grafana" / "dashboards.yml").write_text(
        "apiVersion: 1\n"
        "providers:\n"
        "  - name: LLMOps\n"
        "    folder: LLMOps\n"
        "    type: file\n"
        "    options:\n"
        "      path: /etc/grafana/provisioning/dashboards\n"
    )

    files = [
        str(base / "prometheus.yml"),
        str(base / "alerts.yml"),
        str(base / "grafana" / "dashboards" / "llmops.json"),
        str(base / "grafana" / "datasources" / "prometheus.yml"),
        str(base / "grafana" / "dashboards.yml"),
    ]
    logger.info("Monitoring files written to %s/ (%d files)", output_dir, len(files))
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMOps monitoring utilities")
    parser.add_argument("--generate", action="store_true",
                        help="Write monitoring config files to disk")
    parser.add_argument("--output-dir", default="monitoring")
    parser.add_argument("--demo", action="store_true",
                        help="Print sample metric recording calls")
    args = parser.parse_args()

    if args.generate:
        files = generate_monitoring_files(args.output_dir)
        print(f"Generated {len(files)} files:")
        for f in files:
            print(f"  {f}")

    if args.demo:
        print("\nSample usage:")
        print("  metrics.record_inference(latency_ms=142, prompt_tokens=50,")
        print("                           completion_tokens=200, model='gpt-4o-mini')")
        print("  metrics.record_retrieval('hybrid', latency_ms=23, scores=[0.82, 0.74, 0.61])")
        print("  metrics.set_ragas_scores({'faithfulness': 0.847, 'answer_relevancy': 0.823})")
        print("  metrics.set_ndcg(k=10, value=0.82)")
        print("  metrics.set_model_loaded('llama-3.1-8b', loaded=True)")
        print()
        print("  with metrics.measure('generate', model='gpt-4o-mini'):")
        print("      answer = pipeline.run(query)")

    if not args.generate and not args.demo:
        parser.print_help()
