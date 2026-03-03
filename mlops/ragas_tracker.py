"""
RAGAS Evaluation + MLflow Regression Tracker.
Closes gaps: evaluation pipelines, MLflow, observability, benchmark results
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from mlops.compat import mlflow

logger = logging.getLogger(__name__)


METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class RAGASTracker:
    """
    Production RAG evaluation with regression detection.

    - Runs RAGAS metrics on a question/answer/context dataset
    - Logs all metrics to MLflow
    - Compares against baseline, alerts on regression > threshold
    - Stores historical trends for plotting
    """

    def __init__(
        self,
        experiment_name: str = "rag-evaluation",
        baseline_path: str = "./mlops/ragas_baseline.json",
        regression_threshold: float = 0.05,
    ):
        self.experiment_name = experiment_name
        self.baseline_path = Path(baseline_path)
        self.regression_threshold = regression_threshold
        mlflow.set_experiment(experiment_name)

    def evaluate(
        self, eval_dataset: Dataset, run_name: Optional[str] = None
    ) -> dict:
        """
        Run RAGAS evaluation and log to MLflow.

        eval_dataset must have columns:
          question, answer, contexts, ground_truth (optional)
        """
        logger.info("Running RAGAS eval on %d examples", len(eval_dataset))
        result = evaluate(eval_dataset, metrics=METRICS)
        scores = {
            name: float(result[name])
            for name in METRIC_NAMES
            if name in result
        }

        ts = datetime.now().strftime("%Y%m%d-%H%M")
        with mlflow.start_run(run_name=run_name or f"ragas-eval-{ts}"):
            mlflow.log_metrics(scores)
            mlflow.log_param("num_examples", len(eval_dataset))
            mlflow.log_param("eval_timestamp", datetime.now().isoformat())

            regressions = self._check_regression(scores)
            if regressions:
                logger.warning("REGRESSION DETECTED: %s", regressions)
                mlflow.set_tag("regression_detected", "true")
                mlflow.log_dict(regressions, "regressions.json")
            else:
                mlflow.set_tag("regression_detected", "false")

        self._update_history(scores)
        return scores

    def capture_baseline(self, scores: dict):
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scores": scores,
                    "captured_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        logger.info("Baseline captured: %s", scores)

    def _check_regression(self, current_scores: dict) -> dict:
        if not self.baseline_path.exists():
            logger.info("No baseline found, skipping regression check")
            return {}

        with open(self.baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)["scores"]

        regressions = {}
        for metric in METRIC_NAMES:
            if metric in baseline and metric in current_scores:
                delta = current_scores[metric] - baseline[metric]
                if delta < -self.regression_threshold:
                    regressions[metric] = {
                        "baseline": baseline[metric],
                        "current": current_scores[metric],
                        "delta": delta,
                    }
        return regressions

    def _update_history(self, scores: dict):
        history_path = self.baseline_path.parent / "ragas_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            entry = {"timestamp": datetime.now().isoformat(), **scores}
            f.write(json.dumps(entry) + "\n")

    def plot_trends(self):
        """Plot historical RAGAS metric trends."""
        history_path = self.baseline_path.parent / "ragas_history.jsonl"
        if not history_path.exists():
            logger.warning("No history file found")
            return

        import matplotlib.pyplot as plt

        with open(history_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flatten(), METRIC_NAMES):
            values = [r.get(metric) for r in records]
            ax.plot(range(len(values)), values, marker="o", linewidth=2)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(0, 1)
            ax.axhline(
                y=0.8, color="green", linestyle="--", alpha=0.5, label="Target"
            )
            ax.set_xlabel("Evaluation Run")
            ax.set_ylabel("Score")
            ax.legend()

        plt.suptitle("RAGAS Metric Trends", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output = self.baseline_path.parent / "ragas_trends.png"
        plt.savefig(output, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(output))
        logger.info("Trend plot saved: %s", output)


def build_eval_dataset(qa_pairs: list) -> Dataset:
    """Build RAGAS-compatible dataset from QA pairs."""
    return Dataset.from_dict({
        "question": [qa["question"] for qa in qa_pairs],
        "answer": [qa["answer"] for qa in qa_pairs],
        "contexts": [qa["contexts"] for qa in qa_pairs],
        "ground_truth": [qa.get("ground_truth", "") for qa in qa_pairs],
    })


if __name__ == "__main__":
    tracker = RAGASTracker()
    sample_data = [
        {
            "question": "What is retrieval-augmented generation?",
            "answer": (
                "RAG combines retrieval of relevant documents "
                "with language model generation."
            ),
            "contexts": [
                "RAG is a technique that retrieves relevant passages "
                "then generates answers."
            ],
            "ground_truth": (
                "RAG retrieves documents and uses them to augment "
                "LLM generation."
            ),
        }
    ]
    ds = build_eval_dataset(sample_data)
    scores = tracker.evaluate(ds, run_name="baseline-run")
    tracker.capture_baseline(scores)
    print(f"RAGAS scores: {scores}")
