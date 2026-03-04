"""
causal_inference/counterfactual.py
------------------------------------
Counterfactual evaluation for retrieval pipelines.

Asks: "What would the faithfulness score have been if we had NOT reranked?"
Uses DoWhy's counterfactual inference on observed samples.

Also supports:
  - Minimum chunk count needed to exceed faithfulness threshold
  - What-if: swap top-k chunks and re-score
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualEvaluator:
    """
    Counterfactual reasoning over retrieval pipeline observations.

    Parameters
    ----------
    scoring_fn : callable (df_row -> float) -- your RAGAS faithfulness scorer.
                 If None, uses a linear proxy model fitted on the training data.
    """

    def __init__(self, scoring_fn: Callable | None = None):
        self.scoring_fn = scoring_fn
        self._proxy_model = None
        self._feature_cols: list[str] = []

    def fit_proxy(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        outcome_col: str = "faithfulness",
    ) -> None:
        """
        Fit a linear proxy model for counterfactual scoring when a real
        scorer is not available.
        """
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._proxy_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self._proxy_model.fit(df[feature_cols].values, df[outcome_col].values)
        self._feature_cols = feature_cols
        logger.info("Proxy model fitted on %d samples", len(df))

    def counterfactual_score(
        self,
        df: pd.DataFrame,
        intervention: dict[str, float],
    ) -> pd.DataFrame:
        """
        Compute counterfactual outcomes by applying an intervention.

        Parameters
        ----------
        df           : observed data
        intervention : column -> new_value mapping (e.g. {"reranked": 0})

        Returns
        -------
        DataFrame with original + counterfactual scores + individual treatment effect
        """
        cf_df = df.copy()
        for col, val in intervention.items():
            cf_df[col] = val

        observed_scores = self._score(df)
        cf_scores = self._score(cf_df)

        result = df.copy()
        result["observed_faithfulness"] = observed_scores
        result["cf_faithfulness"] = cf_scores
        result["individual_te"] = observed_scores - cf_scores
        return result

    def what_if_top_k(
        self,
        df: pd.DataFrame,
        k_values: list[int],
        chunk_count_col: str = "num_chunks",
    ) -> pd.DataFrame:
        """
        What-if analysis: how does faithfulness change as we vary top-k retrieved chunks?

        Returns summary DataFrame: k -> mean_faithfulness
        """
        rows = []
        for k in k_values:
            cf = self.counterfactual_score(df, {chunk_count_col: float(k)})
            rows.append({
                "k": k,
                "mean_faithfulness": cf["cf_faithfulness"].mean(),
                "std_faithfulness": cf["cf_faithfulness"].std(),
                "n": len(cf),
            })
        return pd.DataFrame(rows)

    def min_chunks_for_threshold(
        self,
        df: pd.DataFrame,
        threshold: float = 0.85,
        chunk_count_col: str = "num_chunks",
        max_k: int = 20,
    ) -> int | None:
        """
        Find the minimum number of chunks needed so that
        mean counterfactual faithfulness >= threshold.
        """
        summary = self.what_if_top_k(df, list(range(1, max_k + 1)), chunk_count_col)
        passing = summary[summary["mean_faithfulness"] >= threshold]
        if passing.empty:
            logger.warning("Threshold %.2f not reached within k=%d", threshold, max_k)
            return None
        return int(passing["k"].min())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score(self, df: pd.DataFrame) -> np.ndarray:
        if self.scoring_fn is not None:
            return np.array([self.scoring_fn(row) for _, row in df.iterrows()])
        if self._proxy_model is not None:
            return self._proxy_model.predict(df[self._feature_cols].values)
        raise RuntimeError(
            "No scoring function available. Call fit_proxy() or pass scoring_fn."
        )
