"""
Sequential testing with alpha spending for continuous monitoring.

Problem with naive repeated testing:
  If you check p-value every day and stop when p < 0.05, your actual
  false positive rate is much higher than 5% (multiple comparisons).

Solution: alpha spending functions allocate the Type-I error budget
across looks, so the overall false positive rate stays at alpha.

Implementations:
  - O'Brien-Fleming: conservative early, liberal late (best for safety)
  - Pocock: equal alpha at each look (simpler)
  - SPRT: Wald's sequential probability ratio test
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlphaSpending(str, Enum):
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    SPRT = "sprt"


@dataclass
class SequentialTestResult:
    decision: str
    statistic: float
    boundary: float
    alpha_spent: float
    n_looks: int
    n_control: int
    n_treatment: int
    estimated_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float

    def is_significant(self) -> bool:
        return self.decision == "reject_h0"

    def summary(self) -> str:
        return (
            f"Decision: {self.decision}  "
            f"stat={self.statistic:.4f}  boundary={self.boundary:.4f}  "
            f"effect={self.estimated_effect:+.4f}  "
            f"95%CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.p_value:.4f}  looks={self.n_looks}"
        )


class SequentialTest:
    """
    Sequential hypothesis test with alpha spending.

    Usage:
        test = SequentialTest(alpha=0.05, max_n=1000, n_looks=10)
        for batch in data_stream:
            control_obs.extend(batch.control)
            treatment_obs.extend(batch.treatment)
            result = test.update(control_obs, treatment_obs)
            if result.decision != "continue":
                break
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.20,
        max_n: int = 1000,
        n_looks: int = 10,
        spending_fn: AlphaSpending = AlphaSpending.OBRIEN_FLEMING,
        mde: float = 0.02,
    ):
        self.alpha = alpha
        self.beta = beta
        self.max_n = max_n
        self.n_looks = n_looks
        self.spending_fn = spending_fn
        self.mde = mde
        self._look_count = 0
        self._alpha_spent = 0.0

    def update(
        self,
        control: List[float],
        treatment: List[float],
    ) -> SequentialTestResult:
        self._look_count += 1
        n_c, n_t = len(control), len(treatment)
        n_total = n_c + n_t

        if n_c < 5 or n_t < 5:
            return self._continue_result(control, treatment)

        t_stat, p_value = self._welch_t_test(control, treatment)
        boundary = self._compute_boundary(n_total)
        alpha_increment = self._alpha_increment(n_total)
        self._alpha_spent = min(self._alpha_spent + alpha_increment, self.alpha)

        mu_c = sum(control) / n_c
        mu_t = sum(treatment) / n_t
        effect = mu_t - mu_c
        se = self._pooled_se(control, treatment)
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se

        if abs(t_stat) >= boundary:
            decision = "reject_h0"
        elif self._should_accept_null(control, treatment):
            decision = "accept_h0"
        else:
            decision = "continue"

        logger.debug(
            "Sequential test look %d: stat=%.3f boundary=%.3f decision=%s",
            self._look_count, t_stat, boundary, decision,
        )

        return SequentialTestResult(
            decision=decision,
            statistic=t_stat,
            boundary=boundary,
            alpha_spent=self._alpha_spent,
            n_looks=self._look_count,
            n_control=n_c,
            n_treatment=n_t,
            estimated_effect=round(effect, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            p_value=round(p_value, 4),
        )

    def _compute_boundary(self, n_current: int) -> float:
        t = n_current / self.max_n
        t = min(max(t, 0.001), 1.0)

        if self.spending_fn == AlphaSpending.OBRIEN_FLEMING:
            z_alpha = self._norm_quantile(1 - self.alpha / 2)
            return z_alpha / math.sqrt(t)
        elif self.spending_fn == AlphaSpending.POCOCK:
            z_alpha = self._norm_quantile(1 - self.alpha / (2 * self.n_looks))
            return z_alpha
        else:
            return math.log((1 - self.beta) / self.alpha)

    def _alpha_increment(self, n_current: int) -> float:
        t = min(n_current / self.max_n, 1.0)
        if self.spending_fn == AlphaSpending.OBRIEN_FLEMING:
            return 2 * (1 - self._norm_cdf(self._norm_quantile(1 - self.alpha / 2) / math.sqrt(t + 1e-9)))
        return self.alpha / self.n_looks

    def _should_accept_null(self, control: List[float], treatment: List[float]) -> bool:
        n_c, n_t = len(control), len(treatment)
        if n_c + n_t < self.max_n * 0.8:
            return False
        mu_c = sum(control) / n_c
        mu_t = sum(treatment) / n_t
        return abs(mu_t - mu_c) < self.mde / 2

    def _continue_result(self, control: List[float], treatment: List[float]) -> SequentialTestResult:
        return SequentialTestResult(
            decision="continue",
            statistic=0.0,
            boundary=self._compute_boundary(max(len(control) + len(treatment), 1)),
            alpha_spent=self._alpha_spent,
            n_looks=self._look_count,
            n_control=len(control),
            n_treatment=len(treatment),
            estimated_effect=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
        )

    @staticmethod
    def _welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
        n_a, n_b = len(a), len(b)
        mu_a = sum(a) / n_a
        mu_b = sum(b) / n_b
        var_a = sum((x - mu_a) ** 2 for x in a) / max(n_a - 1, 1)
        var_b = sum((x - mu_b) ** 2 for x in b) / max(n_b - 1, 1)
        se = math.sqrt(var_a / n_a + var_b / n_b + 1e-12)
        t = (mu_b - mu_a) / se
        df = max((var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / max(n_a - 1, 1) + (var_b / n_b) ** 2 / max(n_b - 1, 1)
        ), 1.0)
        p = 2 * (1 - SequentialTest._t_cdf(abs(t), df))
        return t, p

    @staticmethod
    def _pooled_se(a: List[float], b: List[float]) -> float:
        n_a, n_b = len(a), len(b)
        mu_a = sum(a) / n_a
        mu_b = sum(b) / n_b
        var_a = sum((x - mu_a) ** 2 for x in a) / max(n_a - 1, 1)
        var_b = sum((x - mu_b) ** 2 for x in b) / max(n_b - 1, 1)
        return math.sqrt(var_a / n_a + var_b / n_b + 1e-12)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_quantile(p: float) -> float:
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0
        a = [2.515517, 0.802853, 0.010328]
        b = [1.432788, 0.189269, 0.001308]
        t = math.sqrt(-2 * math.log(min(p, 1 - p)))
        num = a[0] + a[1] * t + a[2] * t * t
        den = 1 + b[0] * t + b[1] * t * t + b[2] * t * t * t
        x = t - num / den
        return x if p >= 0.5 else -x

    @staticmethod
    def _t_cdf(t: float, df: float) -> float:
        x = df / (df + t * t)
        try:
            from math import lgamma
            log_beta = lgamma(df / 2) + lgamma(0.5) - lgamma((df + 1) / 2)
            if x <= 0 or x >= 1:
                return 0.5
            series = 0.0
            term = 1.0
            for n in range(200):
                term *= x * (df / 2 + n) / ((0.5 + n + 1) * (n + 1) + 1e-12)
                series += term
                if abs(term) < 1e-8:
                    break
            incomplete = math.exp(
                (df / 2) * math.log(x + 1e-12) + 0.5 * math.log(1 - x + 1e-12) - log_beta
            ) * (2 / df + series)
            return min(max(incomplete, 0.0), 1.0)
        except Exception:
            return 0.5
