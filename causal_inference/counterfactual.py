"""
Counterfactual retrieval evaluation.

Answers questions like:
  - "What would faithfulness be if we had NOT used reranking?"
  - "What would faithfulness be with 3 chunks instead of 5?"
  - "What would faithfulness be if context quality were 20% higher?"

Uses the fitted outcome model from the T-Learner to predict potential outcomes
under alternative treatment assignments or covariate interventions.

Usage:
    evaluator = CounterfactualEvaluator()
    evaluator.fit(records)

    # What if reranking was disabled for all queries?
    cf = evaluator.without_reranking()
    print(f"Faithfulness drop: {cf.ate_change:+.4f}")

    # What if we used 3 chunks instead of 5?
    cf = evaluator.with_chunk_count(3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    scenario: str
    observed_mean: float
    counterfactual_mean: float
    ate_change: float
    pct_change: float
    n: int

    def summary(self) -> str:
        direction = "+" if self.ate_change >= 0 else ""
        return (
            f"Counterfactual: {self.scenario}\n"
            f"  Observed:       {self.observed_mean:.4f}\n"
            f"  Counterfactual: {self.counterfactual_mean:.4f}\n"
            f"  Change:         {direction}{self.ate_change:.4f}  "
            f"({direction}{self.pct_change:.1f}%)\n"
            f"  n: {self.n}"
        )


class CounterfactualEvaluator:
    """
    Counterfactual outcome predictor for the retrieval pipeline.

    Fits a T-Learner outcome model on observed data, then uses the
    control-arm model mu_0(x) to predict "what would have happened
    without treatment" for treated units, and vice versa.

    Expected record schema:
        query_complexity: float   [0, 1]
        context_quality:  float   [0, 1]
        chunk_count:      int     [1, 20]
        reranking_used:   int     {0, 1}
        faithfulness:     float   [0, 1]
        answer_relevancy: float   [0, 1]   (optional)
    """

    def __init__(self) -> None:
        self._mu0: object | None = None
        self._mu1: object | None = None
        self._records: list[dict] | None = None
        self._X: np.ndarray | None = None
        self._T: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._feature_cols: list[str] = ["query_complexity", "context_quality", "chunk_count"]

    def fit(self, records: list[dict]) -> "CounterfactualEvaluator":
        from sklearn.ensemble import GradientBoostingRegressor

        self._records = records
        X_list, T_list, Y_list = [], [], []
        for r in records:
            X_list.append([
                r.get("query_complexity", 0.5),
                r.get("context_quality", 0.5),
                r.get("chunk_count", 5),
            ])
            T_list.append(int(r.get("reranking_used", 0)))
            Y_list.append(float(r.get("faithfulness", 0.0)))

        self._X = np.array(X_list)
        self._T = np.array(T_list)
        self._Y = np.array(Y_list)

        mask0 = self._T == 0
        mask1 = self._T == 1

        self._mu0 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self._mu1 = GradientBoostingRegressor(n_estimators=100, random_state=42)

        if mask0.sum() > 1:
            self._mu0.fit(self._X[mask0], self._Y[mask0])
        if mask1.sum() > 1:
            self._mu1.fit(self._X[mask1], self._Y[mask1])

        logger.info(
            "CounterfactualEvaluator fitted: n=%d, n_treated=%d, n_control=%d",
            len(records), mask1.sum(), mask0.sum(),
        )
        return self

    def _check_fitted(self) -> None:
        if self._X is None:
            raise RuntimeError("Call fit() first")

    def without_reranking(self) -> CounterfactualResult:
        """
        For all units, predict what faithfulness would be under no reranking.
        Compares to observed faithfulness.
        """
        self._check_fitted()
        if self._mu0 is None:
            raise RuntimeError("Control model not fitted (no control observations)")

        cf_outcomes = self._mu0.predict(self._X)
        observed_mean = float(self._Y.mean())
        cf_mean = float(cf_outcomes.mean())
        change = cf_mean - observed_mean

        return CounterfactualResult(
            scenario="without reranking (T=0 for all)",
            observed_mean=observed_mean,
            counterfactual_mean=cf_mean,
            ate_change=change,
            pct_change=100 * change / (observed_mean + 1e-9),
            n=len(self._Y),
        )

    def with_reranking(self) -> CounterfactualResult:
        """
        For all units, predict what faithfulness would be if reranking were applied.
        Useful for estimating the gain from universal reranking rollout.
        """
        self._check_fitted()
        if self._mu1 is None:
            raise RuntimeError("Treatment model not fitted (no treated observations)")

        cf_outcomes = self._mu1.predict(self._X)
        observed_mean = float(self._Y.mean())
        cf_mean = float(cf_outcomes.mean())
        change = cf_mean - observed_mean

        return CounterfactualResult(
            scenario="with reranking for all (T=1 for all)",
            observed_mean=observed_mean,
            counterfactual_mean=cf_mean,
            ate_change=change,
            pct_change=100 * change / (observed_mean + 1e-9),
            n=len(self._Y),
        )

    def with_chunk_count(self, target_chunks: int) -> CounterfactualResult:
        """
        Predict faithfulness if chunk_count were set to target_chunks for all queries.
        Uses whichever outcome model matches the current treatment assignment.
        """
        self._check_fitted()

        X_cf = self._X.copy()
        X_cf[:, 2] = float(target_chunks)

        # Use the model that matches each unit's treatment
        cf_outcomes = np.where(
            self._T == 1,
            self._mu1.predict(X_cf) if self._mu1 else self._Y,
            self._mu0.predict(X_cf) if self._mu0 else self._Y,
        )

        observed_mean = float(self._Y.mean())
        cf_mean = float(cf_outcomes.mean())
        change = cf_mean - observed_mean

        return CounterfactualResult(
            scenario=f"chunk_count={target_chunks} for all",
            observed_mean=observed_mean,
            counterfactual_mean=cf_mean,
            ate_change=change,
            pct_change=100 * change / (observed_mean + 1e-9),
            n=len(self._Y),
        )

    def with_context_quality_boost(self, delta: float = 0.2) -> CounterfactualResult:
        """
        Predict faithfulness if context quality were uniformly boosted by delta.
        Models the effect of improving the retrieval index quality.
        """
        self._check_fitted()

        X_cf = self._X.copy()
        X_cf[:, 1] = np.clip(X_cf[:, 1] + delta, 0.0, 1.0)

        cf_outcomes = np.where(
            self._T == 1,
            self._mu1.predict(X_cf) if self._mu1 else self._Y,
            self._mu0.predict(X_cf) if self._mu0 else self._Y,
        )

        observed_mean = float(self._Y.mean())
        cf_mean = float(cf_outcomes.mean())
        change = cf_mean - observed_mean

        return CounterfactualResult(
            scenario=f"context_quality + {delta:.2f} for all",
            observed_mean=observed_mean,
            counterfactual_mean=cf_mean,
            ate_change=change,
            pct_change=100 * change / (observed_mean + 1e-9),
            n=len(self._Y),
        )

    def what_if_report(self) -> str:
        """Run all counterfactual scenarios and return a formatted report."""
        scenarios = [
            self.without_reranking(),
            self.with_reranking(),
            self.with_chunk_count(3),
            self.with_chunk_count(10),
            self.with_context_quality_boost(0.1),
            self.with_context_quality_boost(0.2),
        ]
        lines = ["=" * 60, "Counterfactual What-If Report", "=" * 60]
        for s in scenarios:
            lines.append(s.summary())
            lines.append("")
        return "\n".join(lines)
