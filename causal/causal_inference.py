"""
Causal Inference and Uplift Modeling for LLMOps.

Covers:
  - Uplift modeling: estimate individual treatment effect (ITE)
  - Propensity Score Matching (PSM) to control for selection bias
  - Double ML (Partially Linear Model) for unbiased ATE estimation
  - CUPED variance reduction for A/B experiments
  - Synthetic experiment simulation with RAG retrieval improvements

Real-world framing for LLMOps:
  "Does adding cross-encoder reranking CAUSE higher user satisfaction,
   or do users who see reranked results simply ask better questions?"
  Causal methods answer this. Correlation metrics cannot.

Usage:
    sim = ExperimentSimulator(n_users=1000, true_ate=0.08)
    df = sim.generate()

    estimator = DoubleMLE()
    ate, ci = estimator.estimate(df, treatment="reranking_enabled", outcome="satisfaction")
    print(f"ATE: {ate:.3f}  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    uplift = UpliftModel()
    uplift.fit(df)
    df["predicted_uplift"] = uplift.predict(df)
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalEstimate:
    method: str
    ate: float
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    n_total: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.is_significant() else "not significant"
        return (
            f"{self.method}: ATE={self.ate:+.4f} "
            f"95%CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] "
            f"p={self.p_value:.4f} ({sig})"
        )


# ---------------------------------------------------------------------------
# Experiment simulator
# ---------------------------------------------------------------------------

class ExperimentSimulator:
    """
    Simulate an LLMOps A/B experiment with confounders.

    Generates users with covariates that affect both:
    1. Whether they receive the treatment (selection bias)
    2. The outcome (confounding)

    This is the key challenge causal methods solve:
    naive mean difference overestimates the true treatment effect
    because "power users" are more likely to get reranking AND have
    higher satisfaction regardless.
    """

    def __init__(
        self,
        n_users: int = 1000,
        true_ate: float = 0.08,
        confounding_strength: float = 0.3,
        seed: int = 42,
    ):
        self.n_users = n_users
        self.true_ate = true_ate
        self.confounding_strength = confounding_strength
        random.seed(seed)
        np.random.seed(seed)

    def generate(self) -> List[Dict]:
        """
        Generate synthetic experiment data.

        Covariates (X):
          - query_complexity: how hard the user's queries are
          - session_length: engagement proxy
          - domain_expertise: user domain knowledge level
          - query_count_prior: historical query volume

        Treatment (T): reranking_enabled
          P(T=1|X) = sigmoid(confound * X) — treatment is NOT random (observational data)

        Outcome (Y): user_satisfaction (0-1)
          Y = base_satisfaction(X) + true_ate * T + noise
        """
        records = []
        for i in range(self.n_users):
            query_complexity = np.random.beta(2, 3)
            session_length = np.random.exponential(5)
            domain_expertise = np.random.uniform(0, 1)
            query_count_prior = np.random.poisson(15)

            propensity_score = self._sigmoid(
                self.confounding_strength * (
                    0.4 * query_complexity
                    + 0.3 * (session_length / 10)
                    + 0.3 * domain_expertise
                    - 0.3
                )
            )
            treatment = 1 if random.random() < propensity_score else 0

            base_satisfaction = (
                0.5
                + 0.2 * domain_expertise
                + 0.1 * (session_length / 10)
                - 0.15 * query_complexity
                + np.random.normal(0, 0.05)
            )
            base_satisfaction = float(np.clip(base_satisfaction, 0.0, 1.0))

            individual_te = self.true_ate * (1 + 0.5 * (domain_expertise - 0.5))
            outcome = float(np.clip(
                base_satisfaction + individual_te * treatment + np.random.normal(0, 0.03),
                0.0, 1.0,
            ))

            records.append({
                "user_id": f"u{i:05d}",
                "reranking_enabled": treatment,
                "user_satisfaction": outcome,
                "query_complexity": round(float(query_complexity), 4),
                "session_length": round(float(session_length), 2),
                "domain_expertise": round(float(domain_expertise), 4),
                "query_count_prior": int(query_count_prior),
                "propensity_score": round(float(propensity_score), 4),
                "_true_individual_te": round(float(individual_te), 4),
            })

        return records

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Naive estimator (biased baseline)
# ---------------------------------------------------------------------------

class NaiveDifference:
    """Simple treated-vs-control mean difference. Biased in observational data."""

    def estimate(self, df: List[Dict], treatment: str, outcome: str) -> CausalEstimate:
        treated = [r[outcome] for r in df if r[treatment] == 1]
        control = [r[outcome] for r in df if r[treatment] == 0]

        ate = np.mean(treated) - np.mean(control)
        n_t, n_c = len(treated), len(control)
        pooled_std = math.sqrt(
            (np.var(treated) / n_t + np.var(control) / n_c)
        )
        z = ate / max(pooled_std, 1e-9)
        p_value = 2 * (1 - self._norm_cdf(abs(z)))

        return CausalEstimate(
            method="NaiveDifference",
            ate=float(ate),
            std_err=float(pooled_std),
            ci_lower=float(ate - 1.96 * pooled_std),
            ci_upper=float(ate + 1.96 * pooled_std),
            p_value=float(p_value),
            n_treated=n_t, n_control=n_c, n_total=len(df),
        )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Propensity Score Matching
# ---------------------------------------------------------------------------

class PropensityScoreMatching:
    """
    PSM: match each treated unit to a control unit with similar
    propensity score P(T=1|X), then estimate ATE on matched pairs.

    Removes confounding by ensuring treated and control groups have
    similar covariate distributions after matching.
    """

    def __init__(self, caliper: float = 0.05):
        self.caliper = caliper

    def estimate(
        self,
        df: List[Dict],
        treatment: str,
        outcome: str,
        covariates: List[str],
    ) -> CausalEstimate:
        ps = self._estimate_propensity(df, treatment, covariates)
        for i, rec in enumerate(df):
            rec["_ps"] = ps[i]

        treated = [r for r in df if r[treatment] == 1]
        control = [r for r in df if r[treatment] == 0]
        control_used = set()
        matched_effects = []

        for t_unit in treated:
            best_match = None
            best_dist = float("inf")
            for j, c_unit in enumerate(control):
                if j in control_used:
                    continue
                dist = abs(t_unit["_ps"] - c_unit["_ps"])
                if dist < best_dist and dist < self.caliper:
                    best_dist = dist
                    best_match = j
            if best_match is not None:
                control_used.add(best_match)
                effect = t_unit[outcome] - control[best_match][outcome]
                matched_effects.append(effect)

        if not matched_effects:
            logger.warning("PSM: no matches found within caliper %.3f. Widen caliper.", self.caliper)
            return CausalEstimate("PSM", 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, len(df))

        ate = float(np.mean(matched_effects))
        se = float(np.std(matched_effects) / math.sqrt(len(matched_effects)))
        z = ate / max(se, 1e-9)
        p_value = 2 * (1 - NaiveDifference._norm_cdf(abs(z)))

        return CausalEstimate(
            method="PropensityScoreMatching",
            ate=ate, std_err=se,
            ci_lower=ate - 1.96 * se, ci_upper=ate + 1.96 * se,
            p_value=p_value,
            n_treated=len(matched_effects), n_control=len(matched_effects),
            n_total=len(df),
        )

    def _estimate_propensity(self, df: List[Dict], treatment: str, covariates: List[str]) -> List[float]:
        """Logistic regression for propensity scores (numpy implementation)."""
        X = np.array([[r[c] for c in covariates] for r in df], dtype=float)
        y = np.array([r[treatment] for r in df], dtype=float)

        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_b = np.column_stack([np.ones(len(X_norm)), X_norm])

        w = np.zeros(X_b.shape[1])
        lr = 0.1
        for _ in range(200):
            pred = 1.0 / (1.0 + np.exp(-X_b @ w))
            grad = X_b.T @ (pred - y) / len(y)
            w -= lr * grad

        ps = 1.0 / (1.0 + np.exp(-X_b @ w))
        return ps.tolist()


# ---------------------------------------------------------------------------
# Double ML (Partially Linear Model)
# ---------------------------------------------------------------------------

class DoubleMLE:
    """
    Double Machine Learning (Chernozhukov et al., 2018).

    Debiased ATE estimation via cross-fitting:
      1. Residualise outcome: e_Y = Y - E[Y|X]
      2. Residualise treatment: e_T = T - E[T|X]
      3. Regress e_Y on e_T: coefficient = ATE

    Why this works:
    - Step 1 removes confounding from outcome
    - Step 2 removes selection bias from treatment
    - Regressing residuals gives unbiased ATE even with complex confounders
    - Cross-fitting prevents overfitting bias
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def estimate(
        self,
        df: List[Dict],
        treatment: str,
        outcome: str,
        covariates: List[str],
    ) -> CausalEstimate:
        n = len(df)
        indices = list(range(n))
        random.shuffle(indices)
        folds = [indices[i::self.n_folds] for i in range(self.n_folds)]

        e_Y = np.zeros(n)
        e_T = np.zeros(n)

        for fold_idx, test_idx in enumerate(folds):
            train_idx = [i for j, fold in enumerate(folds) for i in fold if j != fold_idx]

            train = [df[i] for i in train_idx]
            test_data = [df[i] for i in test_idx]

            X_train = np.array([[r[c] for c in covariates] for r in train], dtype=float)
            X_test = np.array([[r[c] for c in covariates] for r in test_data], dtype=float)
            Y_train = np.array([r[outcome] for r in train], dtype=float)
            T_train = np.array([r[treatment] for r in train], dtype=float)

            Y_hat = self._fit_predict_ridge(X_train, Y_train, X_test)
            T_hat = self._fit_predict_logistic(X_train, T_train, X_test)

            for local_idx, global_idx in enumerate(test_idx):
                e_Y[global_idx] = df[global_idx][outcome] - Y_hat[local_idx]
                e_T[global_idx] = df[global_idx][treatment] - T_hat[local_idx]

        ate = float(np.dot(e_T, e_Y) / (np.dot(e_T, e_T) + 1e-12))
        residuals = e_Y - ate * e_T
        se = float(math.sqrt(np.var(residuals) / (np.dot(e_T, e_T) ** 2 / n + 1e-12)))

        z = ate / max(se, 1e-9)
        p_value = float(2 * (1 - NaiveDifference._norm_cdf(abs(z))))

        n_treated = sum(1 for r in df if r[treatment] == 1)
        return CausalEstimate(
            method="DoubleML",
            ate=ate, std_err=se,
            ci_lower=ate - 1.96 * se, ci_upper=ate + 1.96 * se,
            p_value=p_value,
            n_treated=n_treated, n_control=n - n_treated, n_total=n,
        )

    def _fit_predict_ridge(self, X_train, y_train, X_test, alpha: float = 1.0) -> np.ndarray:
        X_n = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        X_t = (X_test - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        Xb = np.column_stack([np.ones(len(X_n)), X_n])
        Xt_b = np.column_stack([np.ones(len(X_t)), X_t])
        A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
        A[0, 0] -= alpha
        w = np.linalg.solve(A, Xb.T @ y_train)
        return Xt_b @ w

    def _fit_predict_logistic(self, X_train, y_train, X_test) -> np.ndarray:
        X_n = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        X_t = (X_test - X_train.mean(0)) / (X_train.std(0) + 1e-8)
        Xb = np.column_stack([np.ones(len(X_n)), X_n])
        Xt_b = np.column_stack([np.ones(len(X_t)), X_t])
        w = np.zeros(Xb.shape[1])
        for _ in range(300):
            pred = 1.0 / (1.0 + np.exp(-Xb @ w))
            grad = Xb.T @ (pred - y_train) / len(y_train)
            w -= 0.1 * grad
        return 1.0 / (1.0 + np.exp(-Xt_b @ w))


# ---------------------------------------------------------------------------
# Uplift Model (T-Learner)
# ---------------------------------------------------------------------------

class UpliftModel:
    """
    T-Learner uplift model: estimate Conditional Average Treatment Effect (CATE).

    Trains two outcome models:
      mu_1(X) = E[Y | T=1, X]  (treated potential outcome)
      mu_0(X) = E[Y | T=0, X]  (control potential outcome)

    Uplift(X) = mu_1(X) - mu_0(X)

    High-uplift users = those who benefit MOST from the intervention.
    Used for: targeted deployment (only enable reranking for users where it helps).
    """

    def __init__(self):
        self._w1: Optional[np.ndarray] = None
        self._w0: Optional[np.ndarray] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._covariates: Optional[List[str]] = None

    def fit(
        self,
        df: List[Dict],
        treatment: str = "reranking_enabled",
        outcome: str = "user_satisfaction",
        covariates: Optional[List[str]] = None,
    ) -> "UpliftModel":
        self._covariates = covariates or ["query_complexity", "session_length", "domain_expertise", "query_count_prior"]

        treated = [r for r in df if r[treatment] == 1]
        control = [r for r in df if r[treatment] == 0]

        X_all = np.array([[r[c] for c in self._covariates] for r in df], dtype=float)
        self._x_mean = X_all.mean(0)
        self._x_std = X_all.std(0) + 1e-8

        X_t = self._normalize(np.array([[r[c] for c in self._covariates] for r in treated], dtype=float))
        y_t = np.array([r[outcome] for r in treated], dtype=float)
        self._w1 = self._fit_ridge(X_t, y_t)

        X_c = self._normalize(np.array([[r[c] for c in self._covariates] for r in control], dtype=float))
        y_c = np.array([r[outcome] for r in control], dtype=float)
        self._w0 = self._fit_ridge(X_c, y_c)

        logger.info("UpliftModel fitted: %d treated, %d control.", len(treated), len(control))
        return self

    def predict(self, df: List[Dict]) -> List[float]:
        if self._w1 is None:
            raise RuntimeError("Call fit() first.")
        X = self._normalize(np.array([[r[c] for c in self._covariates] for r in df], dtype=float))
        Xb = np.column_stack([np.ones(len(X)), X])
        uplift = Xb @ self._w1 - Xb @ self._w0
        return uplift.tolist()

    def top_k_users(self, df: List[Dict], k: int = 100) -> List[Dict]:
        """Return the top-k users who would benefit most from the treatment."""
        uplifts = self.predict(df)
        scored = sorted(zip(uplifts, df), key=lambda x: x[0], reverse=True)
        return [{"uplift": round(u, 4), **r} for u, r in scored[:k]]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        Xb = np.column_stack([np.ones(len(X)), X])
        A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
        A[0, 0] -= alpha
        return np.linalg.solve(A, Xb.T @ y)


# ---------------------------------------------------------------------------
# CUPED variance reduction
# ---------------------------------------------------------------------------

class CUPED:
    """
    Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).

    Reduces variance in A/B experiment outcome by subtracting a
    pre-experiment covariate. Same ATE estimate, tighter confidence intervals,
    smaller required sample size.

    Y_cuped = Y - theta * (X_pre - E[X_pre])
    theta = Cov(Y, X_pre) / Var(X_pre)
    """

    def apply(
        self,
        df: List[Dict],
        treatment: str,
        outcome: str,
        pre_experiment_metric: str,
    ) -> Tuple[float, float, float]:
        """
        Returns: (ate, variance_reduction_pct, new_std_err)
        """
        Y = np.array([r[outcome] for r in df], dtype=float)
        X = np.array([r[pre_experiment_metric] for r in df], dtype=float)
        T = np.array([r[treatment] for r in df], dtype=float)

        theta = float(np.cov(Y, X)[0, 1] / (np.var(X) + 1e-12))
        Y_cuped = Y - theta * (X - X.mean())

        treated_cuped = Y_cuped[T == 1]
        control_cuped = Y_cuped[T == 0]
        ate = float(treated_cuped.mean() - control_cuped.mean())

        var_original = float(np.var(Y[T == 1]) / (T == 1).sum() + np.var(Y[T == 0]) / (T == 0).sum())
        var_cuped = float(np.var(treated_cuped) / len(treated_cuped) + np.var(control_cuped) / len(control_cuped))
        reduction_pct = 100 * (1 - var_cuped / max(var_original, 1e-12))

        return ate, reduction_pct, math.sqrt(var_cuped)


if __name__ == "__main__":
    print("Generating synthetic LLMOps A/B experiment (n=1000, true ATE=0.08)...\n")
    sim = ExperimentSimulator(n_users=1_000, true_ate=0.08, confounding_strength=0.4)
    df = sim.generate()

    covariates = ["query_complexity", "session_length", "domain_expertise", "query_count_prior"]

    naive = NaiveDifference().estimate(df, "reranking_enabled", "user_satisfaction")
    print(naive.summary())

    psm = PropensityScoreMatching(caliper=0.05).estimate(df, "reranking_enabled", "user_satisfaction", covariates)
    print(psm.summary())

    dml = DoubleMLE(n_folds=5).estimate(df, "reranking_enabled", "user_satisfaction", covariates)
    print(dml.summary())

    print(f"\nTrue ATE: 0.0800")
    print(f"Naive bias:  {abs(naive.ate - 0.08):.4f}")
    print(f"PSM bias:    {abs(psm.ate - 0.08):.4f}")
    print(f"DoubleML bias: {abs(dml.ate - 0.08):.4f}")

    uplift = UpliftModel()
    uplift.fit(df)
    predictions = uplift.predict(df)
    print(f"\nUplift model: mean={sum(predictions)/len(predictions):.4f}, "
          f"max={max(predictions):.4f}, min={min(predictions):.4f}")
    top10 = uplift.top_k_users(df, k=10)
    print(f"Top 10 highest-uplift users: avg uplift={sum(u['uplift'] for u in top10)/10:.4f}")

    cuped = CUPED()
    ate_c, var_red, se_c = cuped.apply(df, "reranking_enabled", "user_satisfaction", "query_count_prior")
    print(f"\nCUPED: ATE={ate_c:.4f}, variance reduction={var_red:.1f}%, new SE={se_c:.4f}")
