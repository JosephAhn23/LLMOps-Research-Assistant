"""
causal_inference/uplift.py
--------------------------
Uplift modeling: does cross-encoder reranking *causally* improve RAGAS faithfulness?

Uses DoWhy for causal graph specification + EconML's T-Learner / DR-Learner
to estimate heterogeneous treatment effects (CATE) across query types.

Treatment  : whether cross-encoder reranking was applied (binary)
Outcome    : RAGAS faithfulness score (continuous, 0-1)
Covariates : query length, number of retrieved chunks, query type embedding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UpliftConfig:
    treatment_col: str = "reranked"
    outcome_col: str = "faithfulness"
    covariate_cols: list[str] = field(
        default_factory=lambda: ["query_len", "num_chunks", "query_emb_norm"]
    )
    estimator: Literal["t_learner", "dr_learner", "causal_forest"] = "dr_learner"
    n_estimators: int = 100
    random_state: int = 42


class UpliftEstimator:
    """
    Estimates the causal uplift of retrieval interventions on answer quality.

    Example
    -------
    >>> est = UpliftEstimator()
    >>> results = est.fit_estimate(df)
    >>> print(f"ATE: {results.ate:.4f}  (p={results.p_value:.3f})")
    """

    def __init__(self, config: UpliftConfig | None = None):
        self.config = config or UpliftConfig()
        self._model = None
        self._ate: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_estimate(self, df: pd.DataFrame) -> "UpliftResults":
        """
        Fit causal model and return ATE + CATE estimates.

        Parameters
        ----------
        df : DataFrame with columns matching config (treatment, outcome, covariates)
        """
        self._validate(df)
        T = df[self.config.treatment_col].values.astype(float)
        Y = df[self.config.outcome_col].values.astype(float)
        X = df[self.config.covariate_cols].values.astype(float)

        ate, ate_se, cate = self._estimate(T, Y, X)
        self._ate = ate

        results = UpliftResults(
            ate=ate,
            ate_se=ate_se,
            p_value=self._z_pvalue(ate, ate_se),
            cate=cate,
            covariate_cols=self.config.covariate_cols,
        )
        logger.info(
            "Uplift ATE=%.4f SE=%.4f p=%.4f (estimator=%s)",
            ate, ate_se, results.p_value, self.config.estimator,
        )
        return results

    def top_uplift_segments(
        self, df: pd.DataFrame, results: "UpliftResults", top_k: int = 3
    ) -> pd.DataFrame:
        """Return the top-k covariate segments with highest treatment uplift."""
        seg_df = df[self.config.covariate_cols].copy()
        seg_df["cate"] = results.cate
        seg_df["treatment"] = df[self.config.treatment_col].values

        rows = []
        for col in self.config.covariate_cols:
            seg_df[f"{col}_bin"] = pd.qcut(seg_df[col], q=4, duplicates="drop")
            grp = seg_df.groupby(f"{col}_bin")["cate"].mean().reset_index()
            grp.columns = ["bin", "mean_cate"]
            grp["covariate"] = col
            rows.append(grp)

        summary = pd.concat(rows).sort_values("mean_cate", ascending=False)
        return summary.head(top_k).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """
        Dispatch to chosen estimator. Falls back to simple IPW if
        optional deps (econml) are not installed.
        """
        try:
            return self._econml_estimate(T, Y, X)
        except ImportError:
            logger.warning("econml not installed — falling back to IPW estimator")
            return self._ipw_estimate(T, Y, X)

    def _econml_estimate(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        if self.config.estimator == "t_learner":
            from econml.metalearners import TLearner
            est = TLearner(
                models=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                )
            )
            est.fit(Y, T, X=X)
            cate = est.effect(X).ravel()

        elif self.config.estimator == "dr_learner":
            from econml.dr import DRLearner
            est = DRLearner(
                model_regression=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                ),
                model_propensity=GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                ),
                random_state=self.config.random_state,
            )
            est.fit(Y, T, X=X)
            cate = est.effect(X).ravel()

        else:  # causal_forest
            from econml.grf import CausalForest
            est = CausalForest(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
            )
            est.fit(X, T, Y)
            cate = est.predict(X).ravel()

        self._model = est
        ate = float(cate.mean())
        ate_se = float(cate.std() / np.sqrt(len(cate)))
        return ate, ate_se, cate

    def _ipw_estimate(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Inverse Propensity Weighting — no external deps beyond sklearn."""
        from sklearn.linear_model import LogisticRegression

        ps_model = LogisticRegression(max_iter=500, random_state=self.config.random_state)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1].clip(0.05, 0.95)

        ipw = T * Y / ps - (1 - T) * Y / (1 - ps)
        cate = ipw  # individual-level pseudo-outcomes
        ate = float(ipw.mean())
        ate_se = float(ipw.std() / np.sqrt(len(ipw)))
        return ate, ate_se, cate

    @staticmethod
    def _z_pvalue(ate: float, se: float) -> float:
        from scipy import stats
        if se == 0:
            return 0.0
        return float(2 * (1 - stats.norm.cdf(abs(ate / se))))

    def _validate(self, df: pd.DataFrame) -> None:
        required = (
            [self.config.treatment_col, self.config.outcome_col]
            + self.config.covariate_cols
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


@dataclass
class UpliftResults:
    ate: float
    ate_se: float
    p_value: float
    cate: np.ndarray
    covariate_cols: list[str]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"ATE={self.ate:+.4f}  SE={self.ate_se:.4f}  "
            f"p={self.p_value:.4f}  [{sig}]\n"
            f"CATE range: [{self.cate.min():.4f}, {self.cate.max():.4f}]"
        )
