"""
Sample size and power analysis for A/B experiments.

Answers the question: "How many observations do we need to detect a
meaningful improvement with statistical confidence?"

Key parameters:
  - MDE (Minimum Detectable Effect): smallest effect worth detecting
  - alpha: false positive rate (typically 0.05)
  - power (1-beta): probability of detecting a true effect (typically 0.80)
  - baseline_mean, baseline_std: estimated from historical data

LLMOps example:
  "We want to detect a 2% improvement in RAGAS faithfulness (from 0.847 to 0.867)
   with 80% power and 5% false positive rate. Historical std = 0.08.
   Required n = 502 per variant."
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PowerAnalysisResult:
    required_n_per_variant: int
    required_n_total: int
    mde: float
    alpha: float
    power: float
    baseline_mean: float
    baseline_std: float
    effect_size_cohens_d: float
    days_to_significance: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"Required n per variant: {self.required_n_per_variant:,}",
            f"Required n total: {self.required_n_total:,}",
            f"MDE: {self.mde:+.4f} ({100 * self.mde / max(abs(self.baseline_mean), 1e-9):.1f}% relative)",
            f"Effect size (Cohen's d): {self.effect_size_cohens_d:.3f}",
            f"Alpha: {self.alpha}  Power: {self.power}",
        ]
        if self.days_to_significance:
            lines.append(f"Days to significance (at current traffic): {self.days_to_significance:.1f}")
        return "\n".join(lines)


class PowerAnalysis:
    """
    Sample size calculator and power analysis for A/B experiments.

    Supports:
    - Two-sample t-test (continuous metrics: RAGAS scores, latency)
    - Two-proportion z-test (binary metrics: click rate, error rate)
    - Minimum detectable effect given fixed n
    - Days-to-significance given daily traffic
    """

    def required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> PowerAnalysisResult:
        """
        Compute required sample size per variant for a two-sample t-test.

        Formula: n = 2 * ((z_alpha + z_beta) * sigma / delta)^2
        where delta = MDE, sigma = pooled std
        """
        z_alpha = self._norm_quantile(1 - alpha / (2 if two_sided else 1))
        z_beta = self._norm_quantile(power)

        n = 2 * ((z_alpha + z_beta) * baseline_std / max(abs(mde), 1e-9)) ** 2
        n_per_variant = math.ceil(n)
        cohens_d = abs(mde) / max(baseline_std, 1e-9)

        return PowerAnalysisResult(
            required_n_per_variant=n_per_variant,
            required_n_total=n_per_variant * 2,
            mde=mde,
            alpha=alpha,
            power=power,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            effect_size_cohens_d=round(cohens_d, 3),
        )

    def required_sample_size_proportion(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> PowerAnalysisResult:
        """
        Sample size for binary outcome (click rate, error rate, etc.).
        Uses two-proportion z-test.
        """
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_bar = (p1 + p2) / 2

        z_alpha = self._norm_quantile(1 - alpha / 2)
        z_beta = self._norm_quantile(power)

        n = (
            (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            / max(mde ** 2, 1e-12)
        )
        n_per_variant = math.ceil(n)
        std = math.sqrt(p_bar * (1 - p_bar))

        return PowerAnalysisResult(
            required_n_per_variant=n_per_variant,
            required_n_total=n_per_variant * 2,
            mde=mde,
            alpha=alpha,
            power=power,
            baseline_mean=baseline_rate,
            baseline_std=std,
            effect_size_cohens_d=round(abs(mde) / max(std, 1e-9), 3),
        )

    def minimum_detectable_effect(
        self,
        n_per_variant: int,
        baseline_std: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> float:
        """Given a fixed n, what's the smallest effect we can detect?"""
        z_alpha = self._norm_quantile(1 - alpha / 2)
        z_beta = self._norm_quantile(power)
        return (z_alpha + z_beta) * baseline_std * math.sqrt(2 / max(n_per_variant, 1))

    def achieved_power(
        self,
        n_per_variant: int,
        baseline_std: float,
        true_effect: float,
        alpha: float = 0.05,
    ) -> float:
        """What power do we have given n and a true effect size?"""
        z_alpha = self._norm_quantile(1 - alpha / 2)
        se = baseline_std * math.sqrt(2 / max(n_per_variant, 1))
        z_power = abs(true_effect) / max(se, 1e-9) - z_alpha
        return self._norm_cdf(z_power)

    def days_to_significance(
        self,
        required_n_per_variant: int,
        daily_traffic: int,
        traffic_split: float = 0.5,
    ) -> float:
        """How many days until we have enough observations?"""
        daily_per_variant = daily_traffic * traffic_split
        return required_n_per_variant / max(daily_per_variant, 1)

    def sensitivity_analysis(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde_range: List[float],
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> List[Dict]:
        """Show required n for a range of MDEs."""
        results = []
        for mde in mde_range:
            r = self.required_sample_size(baseline_mean, baseline_std, mde, alpha, power)
            results.append({
                "mde": mde,
                "mde_pct": round(100 * mde / max(abs(baseline_mean), 1e-9), 1),
                "n_per_variant": r.required_n_per_variant,
                "cohens_d": r.effect_size_cohens_d,
            })
        return results

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
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
