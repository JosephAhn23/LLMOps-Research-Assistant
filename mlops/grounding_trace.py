"""
Log grounding / attribution metrics alongside RAG runs (MLflow-compatible).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def log_grounding_metrics(
    grounding_confidence: float,
    *,
    n_chunks: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort MLflow metrics; no-op if no active run."""
    try:
        from mlops.compat import mlflow

        if mlflow.active_run() is None:
            return
        mlflow.log_metric("grounding_confidence", float(grounding_confidence))
        if n_chunks:
            mlflow.log_metric("retrieval_chunks_used", int(n_chunks))
        if extra:
            for k, v in extra.items():
                if isinstance(v, bool):
                    mlflow.log_metric(k, 1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
                elif isinstance(v, str) and len(v) < 500:
                    mlflow.log_param(f"grounding_{k}", v)
    except Exception as exc:
        logger.debug("log_grounding_metrics skipped: %s", exc)
