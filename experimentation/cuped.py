"""
CUPED: Controlled-experiment Using Pre-Experiment Data.

Reference: Deng et al., "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data", KDD 2013.

Key insight:
  If a user's pre-experiment metric (e.g., prior query count) correlates
  with the outcome metric (e.g., satisfaction), we can subtract out that
  correlation to reduce variance — giving tighter confidence intervals
  without collecting more data.

  Y_cuped = Y - theta * (X_pre - E[X_pre])
  theta = Cov(Y, X_pre) / Var(X_pre)

  This is equivalent to running a regression adjustment.
  The ATE estimate is unbiased; only the variance changes.

Variance reduction:
  VR = 1 - rho^2
  where rho = Pearson correlation between Y and X_pre.
  If rho = 0.5, VR = 75% (need 4x fewer observations for same power).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CUPEDResult:
    ate: float
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float
    variance_reduction_pct: float
    theta: float
    correlation: float
    n_treated: int
    n_control: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> str:
        return (
            f"CUPED ATE={self.ate:+.4f}  "
            f"95%CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.p_value:.4f}  "
            f"VR={self.variance_reduction_pct:.1f}%  "
            f"rho={self.correlation:.3f}"
        )


class CUPED:
    """
    CUPED variance reduction for A/B experiments.

    Requires a pre-experiment covariate X that:
    1. Is measured BEFORE the experiment starts (no treatment contamination)
    2. Correlates with the outcome metric Y
    3. Is available for all experiment units

    Typical covariates for LLMOps:
    - Prior query count (correlates with engagement)
    - Prior average latency (correlates with latency outcome)
    - Prior satisfaction score (correlates with satisfaction outcome)
    """

    def apply(
        self,
        treatment_y: List[float],
        control_y: List[float],
        treatment_x_pre: List[float],
        control_x_pre: List[float],
    ) -> CUPEDResult:
        """
        Apply CUPED adjustment and estimate ATE.

        Args:
            treatment_y: outcome metric for treated units
            control_y: outcome metric for control units
            treatment_x_pre: pre-experiment covariate for treated units
            control_x_pre: pre-experiment covariate for control units
        """
        all_y = treatment_y + control_y
        all_x = treatment_x_pre + control_x_pre

        theta = self._compute_theta(all_y, all_x)
        x_mean = sum(all_x) / len(all_x)

        t_cuped = [y - theta * (x - x_mean) for y, x in zip(treatment_y, treatment_x_pre)]
        c_cuped = [y - theta * (x - x_mean) for y, x in zip(control_y, control_x_pre)]

        ate = sum(t_cuped) / len(t_cuped) - sum(c_cuped) / len(c_cuped)

        n_t, n_c = len(t_cuped), len(c_cuped)
        var_t = sum((v - sum(t_cuped) / n_t) ** 2 for v in t_cuped) / max(n_t - 1, 1)
        var_c = sum((v - sum(c_cuped) / n_c) ** 2 for v in c_cuped) / max(n_c - 1, 1)
        se = math.sqrt(var_t / n_t + var_c / n_c + 1e-12)

        z = ate / max(se, 1e-9)
        p_value = 2 * (1 - self._norm_cdf(abs(z)))

        var_original = (
            sum((v - sum(treatment_y) / n_t) ** 2 for v in treatment_y) / max(n_t - 1, 1) / n_t
            + sum((v - sum(control_y) / n_c) ** 2 for v in control_y) / max(n_c - 1, 1) / n_c
        )
        var_cuped = var_t / n_t + var_c / n_c
        variance_reduction = 100 * (1 - var_cuped / max(var_original, 1e-12))

        correlation = self._pearson_correlation(all_y, all_x)

        return CUPEDResult(
            ate=round(ate, 4),
            std_err=round(se, 4),
            ci_lower=round(ate - 1.96 * se, 4),
            ci_upper=round(ate + 1.96 * se, 4),
            p_value=round(p_value, 4),
            variance_reduction_pct=round(variance_reduction, 1),
            theta=round(theta, 4),
            correlation=round(correlation, 4),
            n_treated=n_t,
            n_control=n_c,
        )

    def _compute_theta(self, y: List[float], x: List[float]) -> float:
        n = len(y)
        if n < 2:
            return 0.0
        mu_y = sum(y) / n
        mu_x = sum(x) / n
        cov = sum((yi - mu_y) * (xi - mu_x) for yi, xi in zip(y, x)) / max(n - 1, 1)
        var_x = sum((xi - mu_x) ** 2 for xi in x) / max(n - 1, 1)
        return cov / max(var_x, 1e-12)

    def _pearson_correlation(self, y: List[float], x: List[float]) -> float:
        n = len(y)
        if n < 2:
            return 0.0
        mu_y = sum(y) / n
        mu_x = sum(x) / n
        cov = sum((yi - mu_y) * (xi - mu_x) for yi, xi in zip(y, x))
        std_y = math.sqrt(sum((yi - mu_y) ** 2 for yi in y) + 1e-12)
        std_x = math.sqrt(sum((xi - mu_x) ** 2 for xi in x) + 1e-12)
        return cov / (std_y * std_x)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
