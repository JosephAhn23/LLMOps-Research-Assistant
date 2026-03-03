"""
RAGAS evaluation pipeline for RAG quality measurement.
Covers: MLOps, evaluation pipelines
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from mlops.compat import mlflow

logger = logging.getLogger(__name__)

_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
_REQUIRED_COLUMNS = {"question", "answer", "contexts"}
_MAX_RETRIES = 3


def evaluate_pipeline(test_cases: List[Dict]) -> Dict:
    """
    Run RAGAS evaluation and log scores to MLflow.

    test_cases: [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}]

    Returns a dict of metric name → float score.  Returns an empty dict (and
    logs a warning) if test_cases is empty or all scores are NaN.
    """
    if not test_cases:
        logger.warning("evaluate_pipeline called with empty test_cases — skipping.")
        return {}

    missing = _REQUIRED_COLUMNS - set(test_cases[0].keys())
    if missing:
        raise ValueError(f"test_cases entries are missing required columns: {missing}")

    dataset = Dataset.from_list(test_cases)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            results = evaluate(dataset, metrics=_METRICS)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("RAGAS evaluate attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(
            f"RAGAS evaluation failed after {_MAX_RETRIES} retries"
        ) from last_exc

    raw_scores: Dict[str, float] = results.to_pandas().mean().to_dict()

    # Filter out NaN values — mlflow.log_metrics raises on NaN.
    scores = {k: v for k, v in raw_scores.items() if isinstance(v, float) and not math.isnan(v)}

    if not scores:
        logger.warning("All RAGAS scores are NaN — check that your dataset is valid.")
        return {}

    with mlflow.start_run(run_name="ragas-evaluation"):
        mlflow.log_metrics(scores)
        mlflow.log_param("num_examples", len(test_cases))

    logger.info("RAGAS scores: %s", scores)
    return scores
