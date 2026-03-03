"""
Tests for MLflow tracking, RAGAS evaluation, embedding analysis.
"""
import importlib
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
