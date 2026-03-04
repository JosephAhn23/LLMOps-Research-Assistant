"""
MLflow experiment tracking - retrieval quality, latency, token usage.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable

import mlflow

EXPERIMENT_NAME = "llmops-research-assistant"
_experiment_set = False


def _ensure_experiment() -> None:
    global _experiment_set
    if not _experiment_set:
        mlflow.set_experiment(EXPERIMENT_NAME)
        _experiment_set = True


def track_pipeline(func: Callable) -> Callable:
    """Decorator to auto-track any pipeline function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_experiment()
        with mlflow.start_run(
            run_name=func.__name__,
            nested=mlflow.active_run() is not None,
        ):
            start = time.time()
            result = func(*args, **kwargs)
            latency = time.time() - start

            mlflow.log_metric("latency_seconds", latency)

            if isinstance(result, dict):
                if "tokens_used" in result:
                    mlflow.log_metric("tokens_used", result["tokens_used"])
                if "reranked_chunks" in result:
                    mlflow.log_metric("chunks_retrieved", len(result["reranked_chunks"]))

            return result

    return wrapper


def log_evaluation(query: str, response: str, ground_truth: str, scores: dict):
    """Log RAGAS evaluation scores."""
    _ensure_experiment()
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_param("query", query[:200])
        mlflow.log_metrics(
            {
                "faithfulness": scores.get("faithfulness", 0),
                "answer_relevancy": scores.get("answer_relevancy", 0),
                "context_precision": scores.get("context_precision", 0),
                "context_recall": scores.get("context_recall", 0),
            }
        )
