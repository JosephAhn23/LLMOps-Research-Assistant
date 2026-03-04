"""
Tests for MLflow tracking, RAGAS evaluation, embedding analysis.
"""
import importlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mlflow")


class TestMLflowTracking:
    def test_track_pipeline_logs_latency(self):
        tracking_mod = importlib.import_module("mlops.tracking")
        with patch.object(tracking_mod, "mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            @tracking_mod.track_pipeline
            def dummy_pipeline():
                return {"tokens_used": 100, "reranked_chunks": ["a", "b"]}

            result = dummy_pipeline()
            assert result["tokens_used"] == 100
            mock_mlflow.log_metric.assert_called()

    def test_log_evaluation_logs_all_metrics(self):
        tracking_mod = importlib.import_module("mlops.tracking")
        with patch.object(tracking_mod, "mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            tracking_mod.log_evaluation(
                query="test query",
                response="test response",
                ground_truth="expected answer",
                scores={
                    "faithfulness": 0.91,
                    "answer_relevancy": 0.87,
                    "context_precision": 0.83,
                    "context_recall": 0.79,
                },
            )
            mock_mlflow.log_metrics.assert_called_once()
            logged = mock_mlflow.log_metrics.call_args[0][0]
            assert "faithfulness" in logged
            assert logged["faithfulness"] == 0.91


class TestEvaluatePipeline:
    def test_empty_test_cases_returns_empty_dict(self):
        eval_mod = importlib.import_module("mlops.evaluate")
        result = eval_mod.evaluate_pipeline([])
        assert result == {}

    def test_missing_required_column_raises(self):
        eval_mod = importlib.import_module("mlops.evaluate")
        with pytest.raises(ValueError, match="Missing required columns"):
            eval_mod.evaluate_pipeline([{"question": "q", "answer": "a"}])

    def test_successful_evaluation_logs_to_mlflow(self):
        eval_mod = importlib.import_module("mlops.evaluate")
        fake_scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.75,
        }
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, k: fake_scores[k]
        mock_result.__contains__ = lambda self, k: k in fake_scores

        test_cases = [
            {
                "question": "What is RAG?",
                "answer": "Retrieval-Augmented Generation.",
                "contexts": ["RAG combines retrieval with generation."],
                "ground_truth": "RAG is a technique.",
            }
        ]
        with patch.object(eval_mod, "evaluate", return_value=mock_result), \
             patch.object(eval_mod, "mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = eval_mod.evaluate_pipeline(test_cases)

        assert result["faithfulness"] == 0.9
        mock_mlflow.log_metrics.assert_called_once()

    def test_nan_scores_are_filtered(self):
        import math
        eval_mod = importlib.import_module("mlops.evaluate")
        fake_scores = {
            "faithfulness": float("nan"),
            "answer_relevancy": 0.85,
            "context_precision": float("nan"),
            "context_recall": 0.75,
        }
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, k: fake_scores[k]
        mock_result.__contains__ = lambda self, k: k in fake_scores

        test_cases = [
            {
                "question": "q",
                "answer": "a",
                "contexts": ["ctx"],
            }
        ]
        with patch.object(eval_mod, "evaluate", return_value=mock_result), \
             patch.object(eval_mod, "mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = eval_mod.evaluate_pipeline(test_cases)

        assert "faithfulness" not in result
        assert result.get("answer_relevancy") == 0.85


class TestRAGASTracker:
    def _make_tracker(self, tmp_path: Path):
        tracker_mod = importlib.import_module("mlops.ragas_tracker")
        return tracker_mod.RAGASTracker(
            baseline_path=tmp_path / "baseline.json",
            regression_threshold=0.05,
        )

    def test_capture_baseline_writes_json(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        tracker.capture_baseline(scores)
        data = json.loads((tmp_path / "baseline.json").read_text())
        assert data["scores"]["faithfulness"] == 0.9
        assert "captured_at" in data

    def test_check_regression_detects_drop(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.capture_baseline({"faithfulness": 0.9, "answer_relevancy": 0.85})
        regressions = tracker._check_regression(
            {"faithfulness": 0.8, "answer_relevancy": 0.85}
        )
        assert "faithfulness" in regressions
        assert regressions["faithfulness"]["delta"] == pytest.approx(-0.1, abs=1e-6)

    def test_check_regression_no_regression(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.capture_baseline({"faithfulness": 0.9})
        regressions = tracker._check_regression({"faithfulness": 0.92})
        assert regressions == {}

    def test_check_regression_no_baseline_returns_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        regressions = tracker._check_regression({"faithfulness": 0.9})
        assert regressions == {}

    def test_update_history_appends_jsonl(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker._update_history({"faithfulness": 0.9})
        tracker._update_history({"faithfulness": 0.91})
        history_path = tmp_path / "ragas_history.jsonl"
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["faithfulness"] == 0.9


class TestEmbeddingAnalysis:
    def test_analyze_returns_metrics(self):
        embeddings = np.random.randn(200, 64).astype(np.float32)
        metadata = [{"source": "test", "text": f"text {i}"} for i in range(200)]

        emb_mod = importlib.import_module("analysis.embedding_analysis")
        with patch.object(emb_mod, "mlflow"), patch.object(emb_mod, "KMeans") as mock_km, patch.object(
            emb_mod, "silhouette_score", return_value=0.72
        ):
            mock_km_inst = MagicMock()
            mock_km_inst.fit_predict.return_value = np.zeros(200, dtype=int)
            mock_km.return_value = mock_km_inst
            analyzer = emb_mod.EmbeddingAnalyzer(n_components=2, n_clusters=5)

            with patch.object(analyzer.reducer, "fit_transform", return_value=np.random.randn(200, 2)):
                result = analyzer.analyze(embeddings, metadata)

            assert "metrics" in result
            assert result["metrics"]["n_embeddings"] == 200
            assert result["metrics"]["embedding_dim"] == 64
            assert "silhouette_score" in result["metrics"]
            assert "reduced_embeddings" in result

    def test_intrinsic_dimensionality_reasonable(self):
        """Participation ratio should be between 1 and embedding_dim."""
        dim = 64
        n_samples = 200
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)

        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)

        assert 1.0 <= pr <= dim
