"""
RAGAS evaluation pipeline for RAG quality measurement.
Covers: MLOps, evaluation pipelines
"""
from typing import List, Dict

from datasets import Dataset
import mlflow
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


def evaluate_pipeline(test_cases: List[Dict]) -> Dict:
    """
    test_cases: [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}]
    """
    dataset = Dataset.from_list(test_cases)

    with mlflow.start_run(run_name="ragas-evaluation"):
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = results.to_pandas().mean().to_dict()
        mlflow.log_metrics(scores)
        print("RAGAS Scores:", scores)
        return scores
