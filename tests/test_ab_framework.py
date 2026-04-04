"""
Tests for ABExperiment — variant assignment, run/metrics, and statistical analysis.
"""
from __future__ import annotations

import random
import uuid
from unittest.mock import MagicMock, patch

import pytest

from experiments.ab_framework import ABExperiment, Variant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment(
    *,
    traffic_treatment: float = 0.5,
    primary_metric: str = "answer_relevancy",
    mlflow_tracking: bool = False,
) -> ABExperiment:
    control = Variant("control", lambda: "c", traffic_pct=0.5, description="baseline")
    treatment = Variant("treatment", lambda: "t", traffic_pct=traffic_treatment, description="candidate")
    return ABExperiment(
        experiment_id=str(uuid.uuid4()),
        control=control,
        treatment=treatment,
        primary_metric=primary_metric,
        mlflow_tracking=mlflow_tracking,
    )


# ---------------------------------------------------------------------------
# Variant assignment
# ---------------------------------------------------------------------------


class TestAssignVariant:
    def test_same_user_always_gets_same_variant(self) -> None:
        exp = _make_experiment()
        user_id = str(uuid.uuid4())
        first = exp.assign_variant(user_id)
        second = exp.assign_variant(user_id)
        assert first.name == second.name

    def test_traffic_split_respected_at_scale(self) -> None:
        """10 000 unique users should split within ±5% of declared 50/50."""
        exp = _make_experiment()
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
        exp = _make_experiment()
        variant = exp.assign_variant("user@example.com")
        assert variant.name in ("control", "treatment")


# ---------------------------------------------------------------------------
# Run + observations + analysis
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_raises_when_insufficient_data(self) -> None:
        exp = _make_experiment()
        with pytest.raises(ValueError, match="Insufficient data"):
            exp.analyze()

    def test_analyze_detects_significant_difference(self) -> None:
        """Control at ~0.5, treatment at ~0.9 — should be significant."""
        random.seed(42)
        exp = _make_experiment()

        def metric_low(_: str) -> dict[str, float]:
            return {"answer_relevancy": 0.5 + random.random() * 0.02}

        def metric_high(_: str) -> dict[str, float]:
            return {"answer_relevancy": 0.9 + random.random() * 0.02}

        for _ in range(200):
            rid = str(uuid.uuid4())
            v = exp.assign_variant(rid)
            fn = metric_low if v.name == "control" else metric_high
            exp.run(rid, metric_fn=fn)

        result = exp.analyze()

        assert result.n_control >= 2 and result.n_treatment >= 2
        assert result.significant is True
        assert result.treatment_mean > result.control_mean

    def test_analyze_with_mlflow_tracking_uses_start_run_context(self) -> None:
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        with patch("mlflow.set_experiment"), patch("mlflow.start_run", return_value=mock_run):
            exp = _make_experiment(mlflow_tracking=True)

            def metric_fn(_: str) -> dict[str, float]:
                return {"answer_relevancy": 0.7}

            for i in range(30):
                rid = f"r{i}"
                exp.run(rid, metric_fn=metric_fn)

            result = exp.analyze()

        assert result.experiment_id == exp.experiment_id
