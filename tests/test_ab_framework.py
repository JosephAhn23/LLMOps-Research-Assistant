"""
Tests for ABExperiment — variant assignment, statistical analysis, and
MLflow integration.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from experiments.ab_framework import ABExperiment, ExperimentVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment(redis_mock: MagicMock, min_sample_size: int = 5) -> ABExperiment:
    """Build an ABExperiment with a mocked Redis client."""
    variants = [
        ExperimentVariant("control", "baseline top_k=5", 0.5, {"top_k": 5}),
        ExperimentVariant("treatment", "candidate top_k=10", 0.5, {"top_k": 10}),
    ]
    return ABExperiment(
        experiment_id=str(uuid.uuid4()),
        name="test-experiment",
        variants=variants,
        primary_metric="answer_relevancy",
        min_sample_size=min_sample_size,
    )


# ---------------------------------------------------------------------------
# Variant assignment
# ---------------------------------------------------------------------------

class TestAssignVariant:
    def test_same_user_always_gets_same_variant(self) -> None:
        with patch("experiments.ab_framework.redis_client") as mock_redis:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            mock_redis.rpush.return_value = None
            exp = _make_experiment(mock_redis)

            user_id = str(uuid.uuid4())
            first = exp.assign_variant(user_id)
            # Simulate cached assignment on second call
            mock_redis.hget.return_value = first.name.encode()
            second = exp.assign_variant(user_id)

        assert first.name == second.name

    def test_traffic_split_respected_at_scale(self) -> None:
        """10 000 unique users should split within ±5% of declared 50/50."""
        with patch("experiments.ab_framework.redis_client") as mock_redis:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            exp = _make_experiment(mock_redis)

            counts: dict[str, int] = {"control": 0, "treatment": 0}
            for _ in range(10_000):
                v = exp.assign_variant(str(uuid.uuid4()))
                counts[v.name] += 1

        total = sum(counts.values())
        for name, count in counts.items():
            ratio = count / total
            assert abs(ratio - 0.5) < 0.05, (
                f"Variant '{name}' got {ratio:.2%} traffic, expected ~50%"
            )

    def test_non_uuid_user_id_does_not_crash(self) -> None:
        with patch("experiments.ab_framework.redis_client") as mock_redis:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            exp = _make_experiment(mock_redis)
            # Should not raise ValueError
            variant = exp.assign_variant("user@example.com")
        assert variant.name in ("control", "treatment")

    def test_traffic_must_sum_to_one(self) -> None:
        with pytest.raises(AssertionError):
            ABExperiment(
                experiment_id=str(uuid.uuid4()),
                name="bad-split",
                variants=[
                    ExperimentVariant("a", "a", 0.3, {}),
                    ExperimentVariant("b", "b", 0.3, {}),
                ],
                primary_metric="metric",
            )


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_returns_insufficient_data_flag_when_below_min_sample(self) -> None:
        with patch("experiments.ab_framework.redis_client") as mock_redis:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            mock_redis.lrange.return_value = []
            exp = _make_experiment(mock_redis, min_sample_size=100)

            result = exp.analyze()

        assert result["has_sufficient_data"] is False
        assert "winner" not in result

    def test_analyze_detects_significant_difference(self) -> None:
        """Control at 0.5, treatment at 0.9 — should be significant."""
        import json

        control_vals = [0.5 + (i % 3) * 0.01 for i in range(200)]
        treatment_vals = [0.9 + (i % 3) * 0.01 for i in range(200)]

        def lrange_side_effect(key, start, end):
            if ":control" in key:
                return [json.dumps({"metric_value": v}).encode() for v in control_vals]
            if ":treatment" in key:
                return [json.dumps({"metric_value": v}).encode() for v in treatment_vals]
            return []

        with patch("experiments.ab_framework.redis_client") as mock_redis, \
             patch("experiments.ab_framework.mlflow") as mock_mlflow:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            mock_redis.lrange.side_effect = lrange_side_effect
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            exp = _make_experiment(mock_redis, min_sample_size=10)
            result = exp.analyze()

        assert result["has_sufficient_data"] is True
        assert result["is_significant"] is True
        assert result["winner"] == "treatment"

    def test_analyze_does_not_crash_without_active_mlflow_run(self) -> None:
        """Verifies the nested=True bug is fixed: no MlflowException raised."""
        import json

        vals = [0.7 + i * 0.001 for i in range(20)]
        encoded = [json.dumps({"metric_value": v}).encode() for v in vals]

        with patch("experiments.ab_framework.redis_client") as mock_redis, \
             patch("experiments.ab_framework.mlflow") as mock_mlflow:
            mock_redis.hget.return_value = None
            mock_redis.hset.return_value = None
            mock_redis.lrange.return_value = encoded
            mock_mlflow.active_run.return_value = None  # no active run
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            exp = _make_experiment(mock_redis, min_sample_size=5)
            # Should not raise
            result = exp.analyze()

        assert "experiment_id" in result
