"""
Double Machine Learning for unbiased treatment effect estimation.

Reference: Chernozhukov et al., "Double/Debiased Machine Learning for
Treatment and Structural Parameters", Econometrics Journal, 2018.

Why standard regression fails:
  In observational data, treatment assignment is correlated with confounders.
  A naive OLS regression of Y on T will absorb confounding into the treatment
  coefficient, giving a biased estimate.

Double ML solution (cross-fitting):
  1. Residualise Y: e_Y = Y - E[Y|X]  (remove confounding from outcome)
  2. Residualise T: e_T = T - E[T|X]  (remove selection bias from treatment)
  3. Regress e_Y on e_T: coefficient = unbiased ATE

Cross-fitting (k-fold):
  Use separate folds to fit the nuisance models and compute residuals.
  Prevents regularisation bias from overfitting the nuisance models.

LLMOps application:
  "Does enabling cross-encoder reranking CAUSE higher RAGAS faithfulness,
   or do users who get reranking simply ask better questions?"
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DoubleMLResult:
    ate: float
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_folds: int
    n_total: int
    n_treated: int
    method: str = "DoubleML"

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.is_significant() else "not significant"
        return (
            f"DoubleML ATE={self.ate:+.4f}  "
            f"95%CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.p_value:.4f} ({sig})  "
            f"n={self.n_total} ({self.n_treated} treated)"
        )


class DoubleML:
    """
    Double ML with cross-fitting for unbiased ATE estimation.

    Nuisance models:
    - E[Y|X]: Ridge regression (outcome model)
    - E[T|X]: Logistic regression (propensity model)

    Both are implemented from scratch (numpy only) for portability.
    In production: replace with LightGBM or any sklearn estimator.
    """

    def __init__(self, n_folds: int = 5, alpha_ridge: float = 1.0):
        self.n_folds = n_folds
        self.alpha_ridge = alpha_ridge

    def estimate(
        self,
        Y: List[float],
        T: List[int],
        X: List[List[float]],
    ) -> DoubleMLResult:
        """
        Estimate ATE using Double ML.

        Args:
            Y: outcome variable (e.g., ragas_faithfulness)
            T: binary treatment indicator (0/1)
            X: confounding covariates (e.g., query_complexity, user_expertise)
        """
        import numpy as np

        n = len(Y)
        Y_arr = np.array(Y, dtype=float)
        T_arr = np.array(T, dtype=float)
        X_arr = np.array(X, dtype=float)

        indices = list(range(n))
        random.shuffle(indices)
        folds = [indices[i::self.n_folds] for i in range(self.n_folds)]

        e_Y = np.zeros(n)
        e_T = np.zeros(n)

        for fold_idx, test_idx in enumerate(folds):
            train_idx = [i for j, fold in enumerate(folds) for i in fold if j != fold_idx]

            X_train = X_arr[train_idx]
            X_test = X_arr[test_idx]
            Y_train = Y_arr[train_idx]
            T_train = T_arr[train_idx]

            Y_hat = self._ridge_predict(X_train, Y_train, X_test)
            T_hat = self._logistic_predict(X_train, T_train, X_test)

            for local_i, global_i in enumerate(test_idx):
                e_Y[global_i] = Y_arr[global_i] - Y_hat[local_i]
                e_T[global_i] = T_arr[global_i] - T_hat[local_i]

        ate = float(np.dot(e_T, e_Y) / (np.dot(e_T, e_T) + 1e-12))
        residuals = e_Y - ate * e_T
        se = float(math.sqrt(
            np.var(residuals) / (np.dot(e_T, e_T) ** 2 / n + 1e-12)
        ))

        z = ate / max(se, 1e-9)
        p_value = float(2 * (1 - self._norm_cdf(abs(z))))

        return DoubleMLResult(
            ate=round(ate, 4),
            std_err=round(se, 4),
            ci_lower=round(ate - 1.96 * se, 4),
            ci_upper=round(ate + 1.96 * se, 4),
            p_value=round(p_value, 4),
            n_folds=self.n_folds,
            n_total=n,
            n_treated=int(T_arr.sum()),
        )

    def _ridge_predict(self, X_train, y_train, X_test):
        import numpy as np
        X_n = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        X_t = (X_test - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        Xb = np.column_stack([np.ones(len(X_n)), X_n])
        Xt_b = np.column_stack([np.ones(len(X_t)), X_t])
        A = Xb.T @ Xb + self.alpha_ridge * np.eye(Xb.shape[1])
        A[0, 0] -= self.alpha_ridge
        w = np.linalg.solve(A, Xb.T @ y_train)
        return Xt_b @ w

    def _logistic_predict(self, X_train, y_train, X_test):
        import numpy as np
        X_n = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        X_t = (X_test - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        Xb = np.column_stack([np.ones(len(X_n)), X_n])
        Xt_b = np.column_stack([np.ones(len(X_t)), X_t])
        w = np.zeros(Xb.shape[1])
        for _ in range(300):
            pred = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -20, 20)))
            grad = Xb.T @ (pred - y_train) / len(y_train)
            w -= 0.1 * grad
        return 1.0 / (1.0 + np.exp(-np.clip(Xt_b @ w, -20, 20)))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
