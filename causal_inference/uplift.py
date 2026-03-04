"""
Uplift modeling: does reranking *cause* better RAGAS scores?

Implements:
  - T-Learner: separate outcome models per treatment arm
  - DR-Learner: doubly-robust CATE with cross-fitting (unbiased even if one
    of the nuisance models is misspecified)
  - CausalForest: non-parametric heterogeneous treatment effect estimation
    via econml (optional; falls back to DR-Learner if not installed)

Usage:
    estimator = UpliftEstimator(method="dr_learner")
    estimator.fit(X, T, Y)
    cate = estimator.predict(X_test)
    report = estimator.summary()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UpliftResult:
    method: str
    ate: float
    ate_std: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate: np.ndarray
    high_uplift_fraction: float
    feature_importances: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Uplift Estimation ({self.method})",
            f"  ATE:  {self.ate:+.4f}  [{self.ate_ci_lower:+.4f}, {self.ate_ci_upper:+.4f}]",
            f"  Std:  {self.ate_std:.4f}",
            f"  High-uplift users: {self.high_uplift_fraction:.1%}",
        ]
        if self.feature_importances:
            top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
            lines.append("  Top features: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        return "\n".join(lines)


class TLearner:
    """
    T-Learner: fit separate outcome models mu_0(x) and mu_1(x) for each arm.
    CATE(x) = mu_1(x) - mu_0(x)

    Simple but biased when treatment assignment is correlated with X.
    Use DR-Learner for unbiased estimation.
    """

    def __init__(self) -> None:
        self._mu0: object | None = None
        self._mu1: object | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        from sklearn.ensemble import GradientBoostingRegressor

        mask0 = T == 0
        mask1 = T == 1
        self._mu0 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self._mu1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self._mu0.fit(X[mask0], Y[mask0])
        self._mu1.fit(X[mask1], Y[mask1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mu0 is None or self._mu1 is None:
            raise RuntimeError("Call fit() first")
        return self._mu1.predict(X) - self._mu0.predict(X)


class DRLearner:
    """
    Doubly-Robust Learner with cross-fitting.

    Residualises both outcome and treatment propensity on covariates, then
    estimates CATE on the pseudo-outcome:
        psi(x) = (Y - mu(x)) * (T - e(x)) / (e(x) * (1 - e(x))) + mu_1(x) - mu_0(x)

    Unbiased if *either* the outcome model or the propensity model is correct.
    Cross-fitting prevents overfitting bias from nuisance model estimation.
    """

    def __init__(self, n_folds: int = 5) -> None:
        self.n_folds = n_folds
        self._cate_model: object | None = None
        self._pseudo_outcomes: np.ndarray | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "DRLearner":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        n = len(Y)
        pseudo = np.zeros(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            T_tr, T_val = T[train_idx], T[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            # Nuisance 1: propensity e(x) = P(T=1 | X)
            prop_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            prop_model.fit(X_tr, T_tr)
            e_val = np.clip(prop_model.predict_proba(X_val)[:, 1], 0.05, 0.95)

            # Nuisance 2: outcome mu(x, t) -- separate models per arm
            mu0_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            mu1_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            mask0, mask1 = T_tr == 0, T_tr == 1
            mu0_model.fit(X_tr[mask0], Y_tr[mask0])
            mu1_model.fit(X_tr[mask1], Y_tr[mask1])

            mu0_val = mu0_model.predict(X_val)
            mu1_val = mu1_model.predict(X_val)
            mu_val = T_val * mu1_val + (1 - T_val) * mu0_val

            # DR pseudo-outcome
            pseudo[val_idx] = (
                (Y_val - mu_val) * (T_val - e_val) / (e_val * (1 - e_val))
                + mu1_val - mu0_val
            )

        self._pseudo_outcomes = pseudo
        self._cate_model = Ridge()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._cate_model is None:
            raise RuntimeError("Call fit() first")
        return self._cate_model.predict(X)


def _try_causal_forest(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Attempt econml CausalForest; return (cate, success)."""
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, random_state=42),
            model_t=GradientBoostingClassifier(n_estimators=100, random_state=42),
            n_estimators=200,
            random_state=42,
        )
        cf.fit(Y, T, X=X)
        return cf.effect(X), True
    except ImportError:
        logger.info("econml not installed; falling back to DR-Learner for CausalForest request")
        return np.array([]), False


class UpliftEstimator:
    """
    Unified interface for CATE estimation.

    Parameters
    ----------
    method : "t_learner" | "dr_learner" | "causal_forest"
        Estimation strategy. "causal_forest" requires econml; falls back to
        DR-Learner automatically if econml is not installed.
    """

    def __init__(
        self,
        method: Literal["t_learner", "dr_learner", "causal_forest"] = "dr_learner",
    ) -> None:
        self.method = method
        self._model: TLearner | DRLearner | None = None
        self._cate: np.ndarray | None = None
        self._X: np.ndarray | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "UpliftEstimator":
        """
        Fit the CATE estimator.

        Parameters
        ----------
        X : (n, p) covariate matrix
        T : (n,) binary treatment indicator (0 = control, 1 = treated)
        Y : (n,) continuous outcome (e.g., RAGAS faithfulness score)
        """
        self._X = X
        if self.method == "t_learner":
            self._model = TLearner().fit(X, T, Y)
            self._cate = self._model.predict(X)
        elif self.method == "causal_forest":
            cate, success = _try_causal_forest(X, T, Y)
            if success:
                self._cate = cate
            else:
                self.method = "dr_learner"
                self._model = DRLearner().fit(X, T, Y)
                self._cate = self._model.predict(X)
        else:
            self._model = DRLearner().fit(X, T, Y)
            self._cate = self._model.predict(X)

        logger.info(
            "UpliftEstimator fitted (%s): ATE=%.4f, n=%d",
            self.method, float(self._cate.mean()), len(Y),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return CATE estimates for new observations."""
        if self._model is not None:
            return self._model.predict(X)
        if self._cate is not None:
            return self._cate
        raise RuntimeError("Call fit() first")

    def result(self) -> UpliftResult:
        """Return a structured summary of the estimation."""
        if self._cate is None:
            raise RuntimeError("Call fit() first")

        ate = float(self._cate.mean())
        ate_std = float(self._cate.std() / np.sqrt(len(self._cate)))
        ci_lower = ate - 1.96 * ate_std
        ci_upper = ate + 1.96 * ate_std
        high_uplift = float((self._cate > 0).mean())

        importances: dict[str, float] = {}
        if isinstance(self._model, TLearner) and self._model._mu1 is not None:
            try:
                imp = self._model._mu1.feature_importances_
                importances = {f"feature_{i}": float(v) for i, v in enumerate(imp)}
            except AttributeError:
                pass

        return UpliftResult(
            method=self.method,
            ate=ate,
            ate_std=ate_std,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            cate=self._cate,
            high_uplift_fraction=high_uplift,
            feature_importances=importances,
        )

    def high_uplift_mask(self, threshold: float = 0.0) -> np.ndarray:
        """Boolean mask: True for units with CATE > threshold."""
        if self._cate is None:
            raise RuntimeError("Call fit() first")
        return self._cate > threshold
