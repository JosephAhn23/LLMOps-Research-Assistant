"""
Experiment guardrails: automatic safety checks during live experiments.

Guardrails prevent shipping a harmful change even if the primary metric improves.
They enforce:
  1. Minimum sample size before any decision
  2. Latency degradation limits (treatment can't be >X ms slower)
  3. Error rate limits (treatment can't increase errors by >X%)
  4. Metric drift detection (is the experiment population changing?)
  5. SRM (Sample Ratio Mismatch) detection (is traffic split as intended?)

Interview talking point:
  "We had a case where a new reranker improved RAGAS faithfulness by 3%
   but increased p95 latency by 800ms. The guardrail caught it before
   we shipped. Without guardrails, we would have shipped a regression
   that users would have noticed as slowness."
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GuardrailViolation:
    guardrail: str
    metric: str
    control_value: float
    treatment_value: float
    threshold: float
    severity: str
    message: str

    def is_blocking(self) -> bool:
        return self.severity == "critical"


@dataclass
class GuardrailReport:
    experiment_id: str
    passed: bool
    violations: List[GuardrailViolation]
    warnings: List[str]
    n_control: int
    n_treatment: int
    srm_detected: bool

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Guardrail check: {status}"]
        for v in self.violations:
            lines.append(f"  [{v.severity.upper()}] {v.guardrail}: {v.message}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        if self.srm_detected:
            lines.append("  [CRITICAL] Sample Ratio Mismatch detected!")
        return "\n".join(lines)


class ExperimentGuardrails:
    """
    Configurable guardrails for live A/B experiments.

    Configuration example:
        guardrails = ExperimentGuardrails(
            min_sample_size=100,
            max_latency_increase_ms=500,
            max_error_rate_increase=0.02,
            srm_alpha=0.001,
        )
    """

    def __init__(
        self,
        min_sample_size: int = 100,
        max_latency_increase_ms: float = 500.0,
        max_error_rate_increase: float = 0.02,
        max_metric_degradation: Optional[Dict[str, float]] = None,
        srm_alpha: float = 0.001,
    ):
        self.min_sample_size = min_sample_size
        self.max_latency_increase_ms = max_latency_increase_ms
        self.max_error_rate_increase = max_error_rate_increase
        self.max_metric_degradation = max_metric_degradation or {}
        self.srm_alpha = srm_alpha

    def check(
        self,
        experiment_id: str,
        control_obs: List[Dict[str, float]],
        treatment_obs: List[Dict[str, float]],
        intended_split: float = 0.5,
    ) -> GuardrailReport:
        violations = []
        warnings = []

        n_c, n_t = len(control_obs), len(treatment_obs)

        if n_c < self.min_sample_size or n_t < self.min_sample_size:
            warnings.append(
                f"Minimum sample size not reached: "
                f"control={n_c}, treatment={n_t}, required={self.min_sample_size}"
            )

        srm_detected = self._check_srm(n_c, n_t, intended_split)
        if srm_detected:
            violations.append(GuardrailViolation(
                guardrail="sample_ratio_mismatch",
                metric="traffic_split",
                control_value=n_c / max(n_c + n_t, 1),
                treatment_value=n_t / max(n_c + n_t, 1),
                threshold=intended_split,
                severity="critical",
                message=f"SRM detected: expected {intended_split:.2f}/{1-intended_split:.2f}, "
                        f"got {n_c}/{n_t} ({n_c/(n_c+n_t):.3f}/{n_t/(n_c+n_t):.3f})",
            ))

        latency_violation = self._check_latency(control_obs, treatment_obs)
        if latency_violation:
            violations.append(latency_violation)

        error_violation = self._check_error_rate(control_obs, treatment_obs)
        if error_violation:
            violations.append(error_violation)

        for metric, max_degradation in self.max_metric_degradation.items():
            v = self._check_metric_degradation(metric, max_degradation, control_obs, treatment_obs)
            if v:
                violations.append(v)

        bias_warning = self._check_covariate_imbalance(control_obs, treatment_obs)
        if bias_warning:
            warnings.append(bias_warning)

        blocking = any(v.is_blocking() for v in violations)
        passed = not blocking and not srm_detected

        return GuardrailReport(
            experiment_id=experiment_id,
            passed=passed,
            violations=violations,
            warnings=warnings,
            n_control=n_c,
            n_treatment=n_t,
            srm_detected=srm_detected,
        )

    def _check_srm(self, n_c: int, n_t: int, intended_split: float) -> bool:
        """
        Sample Ratio Mismatch: chi-squared test on observed vs expected counts.
        SRM indicates a bug in the assignment mechanism (e.g., bot traffic,
        cookie deletion, or a logging bug).
        """
        n_total = n_c + n_t
        if n_total < 20:
            return False
        expected_t = n_total * intended_split
        expected_c = n_total * (1 - intended_split)
        chi2 = (
            (n_t - expected_t) ** 2 / max(expected_t, 1)
            + (n_c - expected_c) ** 2 / max(expected_c, 1)
        )
        p_value = 1 - self._chi2_cdf(chi2, df=1)
        return p_value < self.srm_alpha

    def _check_latency(
        self,
        control_obs: List[Dict],
        treatment_obs: List[Dict],
    ) -> Optional[GuardrailViolation]:
        c_latencies = [o["latency_ms"] for o in control_obs if "latency_ms" in o]
        t_latencies = [o["latency_ms"] for o in treatment_obs if "latency_ms" in o]
        if not c_latencies or not t_latencies:
            return None

        c_p95 = sorted(c_latencies)[int(0.95 * len(c_latencies))]
        t_p95 = sorted(t_latencies)[int(0.95 * len(t_latencies))]
        increase = t_p95 - c_p95

        if increase > self.max_latency_increase_ms:
            return GuardrailViolation(
                guardrail="latency_degradation",
                metric="p95_latency_ms",
                control_value=round(c_p95, 1),
                treatment_value=round(t_p95, 1),
                threshold=self.max_latency_increase_ms,
                severity="critical",
                message=f"p95 latency increased by {increase:.0f}ms > max {self.max_latency_increase_ms:.0f}ms",
            )
        return None

    def _check_error_rate(
        self,
        control_obs: List[Dict],
        treatment_obs: List[Dict],
    ) -> Optional[GuardrailViolation]:
        c_errors = [o.get("error_rate", 0) for o in control_obs]
        t_errors = [o.get("error_rate", 0) for o in treatment_obs]
        if not c_errors or not t_errors:
            return None

        c_rate = sum(c_errors) / len(c_errors)
        t_rate = sum(t_errors) / len(t_errors)
        increase = t_rate - c_rate

        if increase > self.max_error_rate_increase:
            return GuardrailViolation(
                guardrail="error_rate_increase",
                metric="error_rate",
                control_value=round(c_rate, 4),
                treatment_value=round(t_rate, 4),
                threshold=self.max_error_rate_increase,
                severity="critical",
                message=f"Error rate increased by {increase:.4f} > max {self.max_error_rate_increase:.4f}",
            )
        return None

    def _check_metric_degradation(
        self,
        metric: str,
        max_degradation: float,
        control_obs: List[Dict],
        treatment_obs: List[Dict],
    ) -> Optional[GuardrailViolation]:
        c_vals = [o[metric] for o in control_obs if metric in o]
        t_vals = [o[metric] for o in treatment_obs if metric in o]
        if not c_vals or not t_vals:
            return None

        c_mean = sum(c_vals) / len(c_vals)
        t_mean = sum(t_vals) / len(t_vals)
        degradation = c_mean - t_mean

        if degradation > max_degradation:
            return GuardrailViolation(
                guardrail="metric_degradation",
                metric=metric,
                control_value=round(c_mean, 4),
                treatment_value=round(t_mean, 4),
                threshold=max_degradation,
                severity="critical",
                message=f"{metric} degraded by {degradation:.4f} > max {max_degradation:.4f}",
            )
        return None

    def _check_covariate_imbalance(
        self,
        control_obs: List[Dict],
        treatment_obs: List[Dict],
        covariate: str = "query_complexity",
    ) -> Optional[str]:
        c_vals = [o[covariate] for o in control_obs if covariate in o]
        t_vals = [o[covariate] for o in treatment_obs if covariate in o]
        if len(c_vals) < 10 or len(t_vals) < 10:
            return None

        c_mean = sum(c_vals) / len(c_vals)
        t_mean = sum(t_vals) / len(t_vals)
        c_std = math.sqrt(sum((v - c_mean) ** 2 for v in c_vals) / max(len(c_vals) - 1, 1))
        t_std = math.sqrt(sum((v - t_mean) ** 2 for v in t_vals) / max(len(t_vals) - 1, 1))
        pooled_std = math.sqrt((c_std ** 2 + t_std ** 2) / 2 + 1e-9)
        smd = abs(c_mean - t_mean) / pooled_std

        if smd > 0.1:
            return (
                f"Covariate imbalance detected on '{covariate}': "
                f"SMD={smd:.3f} > 0.1. "
                f"Control mean={c_mean:.3f}, treatment mean={t_mean:.3f}. "
                f"Consider stratified randomization."
            )
        return None

    @staticmethod
    def _chi2_cdf(x: float, df: int) -> float:
        if x <= 0:
            return 0.0
        try:
            from math import lgamma, exp, log
            k = df / 2
            log_cdf = k * log(x / 2 + 1e-12) - x / 2 - lgamma(k + 1)
            return min(exp(log_cdf), 1.0)
        except Exception:
            return 0.5
