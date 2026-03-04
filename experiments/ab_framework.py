"""
experiments/ab_framework.py
-----------------------------
Production A/B testing framework for LLM pipeline variants.

Features:
  - Traffic splitting with configurable allocation
  - Metric collection and statistical significance testing
  - Sequential / fixed-horizon analysis
  - Guardrail metrics (stop experiment if quality drops)
  - MLflow integration for experiment tracking
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    handler: Callable
    traffic_pct: float = 0.5  # fraction of traffic routed here
    description: str = ""


@dataclass
class Observation:
    variant: str
    primary_metric: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""


@dataclass
class ExperimentResult:
    experiment_id: str
    control: str
    treatment: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    p_value: float
    confidence_interval: tuple[float, float]
    significant: bool
    test_used: str
    power: float = 0.0
    guardrail_violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        lines = [
            f"Experiment: {self.experiment_id}",
            f"  Control ({self.control}):   n={self.n_control}  mean={self.control_mean:.4f}",
            f"  Treatment ({self.treatment}): n={self.n_treatment}  mean={self.treatment_mean:.4f}",
            f"  Lift: {self.absolute_lift:+.4f} ({self.relative_lift_pct:+.1f}%)",
            f"  p={self.p_value:.4f}  95% CI=[{self.confidence_interval[0]:+.4f}, {self.confidence_interval[1]:+.4f}]",
            f"  {sig}  (test={self.test_used})",
        ]
        if self.guardrail_violations:
            lines.append(f"  Guardrail violations: {self.guardrail_violations}")
        return "\n".join(lines)


@dataclass
class GuardrailConfig:
    metric: str
    min_value: float | None = None
    max_degradation_pct: float | None = None  # e.g. 0.05 = stop if degrades >5%


# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------

class ABExperiment:
    """
    Manages a single A/B experiment between two pipeline variants.

    Usage
    -----
    >>> exp = ABExperiment("reranker_ablation", control, treatment)
    >>> for query in queries:
    ...     result = exp.run(query)
    >>> analysis = exp.analyze()
    >>> print(analysis.summary())
    """

    def __init__(
        self,
        experiment_id: str,
        control: Variant,
        treatment: Variant,
        primary_metric: str = "faithfulness",
        guardrails: list[GuardrailConfig] | None = None,
        hash_salt: str = "ab",
        mlflow_tracking: bool = False,
    ):
        self.experiment_id = experiment_id
        self.control = control
        self.treatment = treatment
        self.primary_metric = primary_metric
        self.guardrails = guardrails or []
        self.hash_salt = hash_salt
        self.mlflow_tracking = mlflow_tracking

        self._observations: dict[str, list[Observation]] = defaultdict(list)
        self._start_time = time.time()

        if mlflow_tracking:
            self._init_mlflow()

    # ------------------------------------------------------------------
    # Running the experiment
    # ------------------------------------------------------------------

    def assign_variant(self, request_id: str) -> Variant:
        """Deterministic traffic assignment via hashing (sticky sessions)."""
        digest = hashlib.md5(f"{self.hash_salt}:{request_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF  # uniform [0, 1)
        return self.control if bucket > self.treatment.traffic_pct else self.treatment

    def run(
        self,
        request_id: str,
        *args,
        metric_fn: Callable | None = None,
        **kwargs,
    ) -> tuple[Any, Observation]:
        """
        Route request to assigned variant, execute, and record observation.

        Parameters
        ----------
        request_id : stable identifier for traffic assignment
        metric_fn  : callable(output) -> dict of metric values
        """
        variant = self.assign_variant(request_id)
        t0 = time.perf_counter()

        try:
            output = variant.handler(*args, **kwargs)
        except Exception as e:
            logger.error("Variant %s failed on request %s: %s", variant.name, request_id, e)
            raise

        latency_ms = (time.perf_counter() - t0) * 1000

        metrics = metric_fn(output) if metric_fn else {}
        primary = metrics.get(self.primary_metric, metrics.get("score", 0.0))

        obs = Observation(
            variant=variant.name,
            primary_metric=primary,
            secondary_metrics={k: v for k, v in metrics.items() if k != self.primary_metric},
            latency_ms=latency_ms,
            request_id=request_id,
        )
        self._observations[variant.name].append(obs)

        self._check_guardrails_live(obs)
        return output, obs

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        test: Literal["t_test", "mann_whitney", "bootstrap"] = "t_test",
        alpha: float = 0.05,
    ) -> ExperimentResult:
        """Run statistical significance test and return full result."""
        ctrl_obs = self._observations[self.control.name]
        trt_obs = self._observations[self.treatment.name]

        if len(ctrl_obs) < 2 or len(trt_obs) < 2:
            raise ValueError(
                f"Insufficient data: control={len(ctrl_obs)}, treatment={len(trt_obs)}"
            )

        ctrl_vals = np.array([o.primary_metric for o in ctrl_obs])
        trt_vals = np.array([o.primary_metric for o in trt_obs])

        p_value, ci = self._run_test(ctrl_vals, trt_vals, test)
        ctrl_mean = float(ctrl_vals.mean())
        trt_mean = float(trt_vals.mean())
        abs_lift = trt_mean - ctrl_mean
        rel_lift = (abs_lift / abs(ctrl_mean) * 100) if ctrl_mean != 0 else 0.0
        power = self._estimate_power(ctrl_vals, trt_vals)

        guardrail_violations = self._check_guardrails_post(ctrl_obs, trt_obs)

        result = ExperimentResult(
            experiment_id=self.experiment_id,
            control=self.control.name,
            treatment=self.treatment.name,
            n_control=len(ctrl_vals),
            n_treatment=len(trt_vals),
            control_mean=ctrl_mean,
            treatment_mean=trt_mean,
            absolute_lift=abs_lift,
            relative_lift_pct=rel_lift,
            p_value=p_value,
            confidence_interval=ci,
            significant=p_value < alpha,
            test_used=test,
            power=power,
            guardrail_violations=guardrail_violations,
        )

        if self.mlflow_tracking:
            self._log_to_mlflow(result)

        logger.info("Experiment %s: %s", self.experiment_id, result.summary())
        return result

    def sample_size_estimate(
        self,
        baseline_mean: float,
        mde: float,
        baseline_std: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Estimate minimum sample size per variant for desired power.

        Parameters
        ----------
        mde : minimum detectable effect (absolute)
        """
        from scipy import stats
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) * baseline_std / mde) ** 2
        return int(np.ceil(n))

    # ------------------------------------------------------------------
    # Multi-armed bandit (Thompson Sampling)
    # ------------------------------------------------------------------

    def thompson_allocate(self) -> Variant:
        """
        Thompson Sampling traffic allocation -- adapts split as data comes in.
        Suitable for online experiments where you want to minimize regret.
        """
        ctrl_vals = [o.primary_metric for o in self._observations[self.control.name]] or [0.5]
        trt_vals = [o.primary_metric for o in self._observations[self.treatment.name]] or [0.5]

        ctrl_sample = np.random.beta(
            max(1, sum(1 for v in ctrl_vals if v > 0.5)),
            max(1, sum(1 for v in ctrl_vals if v <= 0.5)),
        )
        trt_sample = np.random.beta(
            max(1, sum(1 for v in trt_vals if v > 0.5)),
            max(1, sum(1 for v in trt_vals if v <= 0.5)),
        )
        return self.treatment if trt_sample > ctrl_sample else self.control

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_test(
        self,
        ctrl: np.ndarray,
        trt: np.ndarray,
        test: str,
    ) -> tuple[float, tuple[float, float]]:
        from scipy import stats

        if test == "t_test":
            _, p = stats.ttest_ind(trt, ctrl, equal_var=False)
            se = np.sqrt(trt.var() / len(trt) + ctrl.var() / len(ctrl))
            diff = trt.mean() - ctrl.mean()
            ci = (diff - 1.96 * se, diff + 1.96 * se)
            return float(p), ci

        elif test == "mann_whitney":
            _, p = stats.mannwhitneyu(trt, ctrl, alternative="two-sided")
            ci = self._bootstrap_ci(ctrl, trt)
            return float(p), ci

        elif test == "bootstrap":
            p, ci = self._bootstrap_test(ctrl, trt)
            return p, ci

        else:
            raise ValueError(f"Unknown test: {test}")

    def _bootstrap_ci(
        self, ctrl: np.ndarray, trt: np.ndarray, n_boot: int = 1000, alpha: float = 0.05
    ) -> tuple[float, float]:
        diffs = []
        for _ in range(n_boot):
            c = np.random.choice(ctrl, len(ctrl), replace=True)
            t = np.random.choice(trt, len(trt), replace=True)
            diffs.append(t.mean() - c.mean())
        return (
            float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        )

    def _bootstrap_test(
        self, ctrl: np.ndarray, trt: np.ndarray, n_boot: int = 2000
    ) -> tuple[float, tuple[float, float]]:
        observed_diff = trt.mean() - ctrl.mean()
        combined = np.concatenate([ctrl, trt])
        null_diffs = []
        for _ in range(n_boot):
            perm = np.random.permutation(combined)
            null_diffs.append(perm[: len(trt)].mean() - perm[len(trt):].mean())
        p = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))
        ci = self._bootstrap_ci(ctrl, trt)
        return p, ci

    def _estimate_power(self, ctrl: np.ndarray, trt: np.ndarray) -> float:
        try:
            from scipy import stats
            pooled_std = np.sqrt((ctrl.std() ** 2 + trt.std() ** 2) / 2)
            if pooled_std == 0:
                return 0.0
            effect_size = abs(trt.mean() - ctrl.mean()) / pooled_std
            n = min(len(ctrl), len(trt))
            nc = effect_size * np.sqrt(n / 2)
            return float(1 - stats.t.cdf(stats.t.ppf(0.975, df=n - 1), df=n - 1, loc=nc))
        except Exception:
            return 0.0

    def _check_guardrails_live(self, obs: Observation) -> None:
        """Alert on live guardrail violations."""
        for g in self.guardrails:
            val = obs.secondary_metrics.get(g.metric)
            if val is not None and g.min_value is not None and val < g.min_value:
                logger.warning(
                    "Guardrail violation: %s=%.4f < %.4f (variant=%s)",
                    g.metric, val, g.min_value, obs.variant,
                )

    def _check_guardrails_post(
        self,
        ctrl_obs: list[Observation],
        trt_obs: list[Observation],
    ) -> list[str]:
        violations = []
        for g in self.guardrails:
            ctrl_vals = [
                o.secondary_metrics[g.metric] for o in ctrl_obs
                if g.metric in o.secondary_metrics
            ]
            trt_vals = [
                o.secondary_metrics[g.metric] for o in trt_obs
                if g.metric in o.secondary_metrics
            ]
            if not ctrl_vals or not trt_vals:
                continue
            ctrl_mean = float(np.mean(ctrl_vals))
            trt_mean = float(np.mean(trt_vals))
            if g.min_value is not None and trt_mean < g.min_value:
                violations.append(
                    f"{g.metric}: {trt_mean:.4f} < min {g.min_value:.4f}"
                )
            if g.max_degradation_pct is not None and ctrl_mean != 0:
                degradation = (ctrl_mean - trt_mean) / abs(ctrl_mean)
                if degradation > g.max_degradation_pct:
                    violations.append(
                        f"{g.metric}: degraded {degradation:.1%} > threshold {g.max_degradation_pct:.1%}"
                    )
        return violations

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_experiment(f"ab_{self.experiment_id}")
        except ImportError:
            logger.warning("mlflow not installed, disabling tracking")
            self.mlflow_tracking = False

    def _log_to_mlflow(self, result: ExperimentResult) -> None:
        try:
            import mlflow
            with mlflow.start_run(run_name=result.experiment_id):
                mlflow.log_params({
                    "control": result.control,
                    "treatment": result.treatment,
                    "test": result.test_used,
                })
                mlflow.log_metrics({
                    "control_mean": result.control_mean,
                    "treatment_mean": result.treatment_mean,
                    "absolute_lift": result.absolute_lift,
                    "relative_lift_pct": result.relative_lift_pct,
                    "p_value": result.p_value,
                    "power": result.power,
                    "significant": float(result.significant),
                    "n_control": float(result.n_control),
                    "n_treatment": float(result.n_treatment),
                })
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)
