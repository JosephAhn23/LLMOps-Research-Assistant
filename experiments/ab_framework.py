"""
A/B experimentation framework with statistical significance testing.
Covers: Experimentation frameworks
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import hashlib
import os

from mlops.compat import mlflow
import numpy as np
import redis
from scipy import stats

redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


class ExperimentStatus(Enum):
    RUNNING = "running"
    CONCLUDED = "concluded"
    PAUSED = "paused"


@dataclass
class ExperimentVariant:
    name: str
    description: str
    traffic_pct: float  # 0.0 to 1.0
    config: Dict


@dataclass
class ExperimentResult:
    variant: str
    metric_value: float
    timestamp: float
    request_id: str
    metadata: Dict = field(default_factory=dict)


class ABExperiment:
    """
    Statistical A/B testing for RAG pipeline variants.
    Covers: Experimentation frameworks, statistical significance
    """

    def __init__(
        self,
        experiment_id: str,
        name: str,
        variants: List[ExperimentVariant],
        primary_metric: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05,
        min_detectable_effect: float = 0.05,
    ):
        assert abs(sum(v.traffic_pct for v in variants) - 1.0) < 0.001, "Variant traffic must sum to 1.0"

        self.experiment_id = experiment_id
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.primary_metric = primary_metric
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.min_detectable_effect = min_detectable_effect
        self.status = ExperimentStatus.RUNNING

        _ttl = 86400 * 30  # 30 days
        redis_client.hset(
            f"experiment:{experiment_id}",
            mapping={
                "name": name,
                "status": self.status.value,
                "primary_metric": primary_metric,
                "created_at": str(time.time()),
                "config": json.dumps(
                    {
                        "variants": [
                            {"name": v.name, "traffic_pct": v.traffic_pct, "config": v.config} for v in variants
                        ]
                    }
                ),
            },
        )
        redis_client.expire(f"experiment:{experiment_id}", _ttl)
        redis_client.expire(f"experiment:{experiment_id}:assignments", _ttl)

    def assign_variant(self, user_id: str) -> ExperimentVariant:
        """
        Deterministic variant assignment via hashing.
        Same user always gets same variant.
        """
        cached = redis_client.hget(f"experiment:{self.experiment_id}:assignments", user_id)
        if cached:
            variant_name = cached.decode()
            return self.variants[variant_name]

        try:
            hash_val = int(uuid.UUID(user_id).int % 10000) / 10000
        except ValueError:
            # user_id is not a UUID — fall back to MD5 hash for stable assignment.
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 10000 / 10000

        cumulative = 0.0
        assigned_variant = list(self.variants.values())[-1]
        for variant in self.variants.values():
            cumulative += variant.traffic_pct
            if hash_val < cumulative:
                assigned_variant = variant
                break

        redis_client.hset(f"experiment:{self.experiment_id}:assignments", user_id, assigned_variant.name)
        return assigned_variant

    def record_result(
        self,
        variant_name: str,
        metric_value: float,
        request_id: str,
        metadata: Optional[Dict] = None,
    ):
        """Record metric observation for a variant."""
        result = ExperimentResult(
            variant=variant_name,
            metric_value=metric_value,
            timestamp=time.time(),
            request_id=request_id,
            metadata=metadata or {},
        )
        results_key = f"experiment:{self.experiment_id}:results:{variant_name}"
        redis_client.rpush(
            results_key,
            json.dumps(
                {
                    "metric_value": metric_value,
                    "timestamp": result.timestamp,
                    "request_id": request_id,
                    "metadata": result.metadata,
                }
            ),
        )
        redis_client.expire(results_key, 86400 * 30)

    def get_results(self, variant_name: str) -> List[float]:
        """Get all recorded metric values for a variant."""
        raw = redis_client.lrange(f"experiment:{self.experiment_id}:results:{variant_name}", 0, -1)
        return [json.loads(r)["metric_value"] for r in raw]

    def analyze(self) -> Dict:
        """
        Statistical analysis: t-test, effect size, confidence intervals.
        Determines if experiment has reached significance.
        """
        variant_names = list(self.variants.keys())
        if len(variant_names) < 2:
            return {"error": "need at least 2 variants"}

        control_name = variant_names[0]
        treatment_name = variant_names[1]
        control_values = self.get_results(control_name)
        treatment_values = self.get_results(treatment_name)
        n_control = len(control_values)
        n_treatment = len(treatment_values)

        has_sufficient_data = n_control >= self.min_sample_size and n_treatment >= self.min_sample_size

        result = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "primary_metric": self.primary_metric,
            "status": self.status.value,
            "variants": {
                control_name: {
                    "n": n_control,
                    "mean": round(np.mean(control_values), 4) if control_values else None,
                    "std": round(np.std(control_values), 4) if control_values else None,
                    "ci_95": self._confidence_interval(control_values),
                },
                treatment_name: {
                    "n": n_treatment,
                    "mean": round(np.mean(treatment_values), 4) if treatment_values else None,
                    "std": round(np.std(treatment_values), 4) if treatment_values else None,
                    "ci_95": self._confidence_interval(treatment_values),
                },
            },
            "has_sufficient_data": has_sufficient_data,
        }

        if has_sufficient_data:
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            pooled_std = np.sqrt((np.std(control_values) ** 2 + np.std(treatment_values) ** 2) / 2)
            cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / max(pooled_std, 1e-8)
            relative_lift = (np.mean(treatment_values) - np.mean(control_values)) / max(
                abs(np.mean(control_values)), 1e-8
            )

            is_significant = p_value < self.significance_level
            winner = (
                treatment_name
                if (is_significant and cohens_d > 0)
                else control_name
                if (is_significant and cohens_d < 0)
                else None
            )

            result.update(
                {
                    "statistical_test": "two_sample_t_test",
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 6),
                    "cohens_d": round(cohens_d, 4),
                    "relative_lift": round(relative_lift, 4),
                    "is_significant": is_significant,
                    "significance_level": self.significance_level,
                    "winner": winner,
                    "recommendation": self._get_recommendation(is_significant, winner, cohens_d),
                }
            )

            with mlflow.start_run(
                run_name=f"ab-analysis-{self.experiment_id}",
                nested=mlflow.active_run() is not None,
            ):
                mlflow.log_metrics(
                    {
                        "p_value": p_value,
                        "cohens_d": abs(cohens_d),
                        "relative_lift": relative_lift,
                        f"{control_name}_mean": np.mean(control_values),
                        f"{treatment_name}_mean": np.mean(treatment_values),
                    }
                )
                mlflow.log_param("winner", winner or "inconclusive")

        return result

    def _confidence_interval(self, values: List[float], confidence: float = 0.95):
        if len(values) < 2:
            return None
        mean = np.mean(values)
        se = stats.sem(values)
        h = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        return [round(mean - h, 4), round(mean + h, 4)]

    def _get_recommendation(self, is_significant: bool, winner: Optional[str], cohens_d: float) -> str:
        if not is_significant:
            return "Continue collecting data - no significant difference detected yet"
        if abs(cohens_d) < self.min_detectable_effect:
            return "Significant but effect size too small - practical difference negligible"
        if winner:
            return f"Ship {winner} - statistically significant improvement (Cohen's d={cohens_d:.3f})"
        return "Inconclusive"


class ExperimentRegistry:
    """Central registry for all running experiments."""

    def __init__(self):
        self.experiments: Dict[str, ABExperiment] = {}

    def create_rag_retrieval_experiment(self) -> ABExperiment:
        """Example: Test top_k=5 vs top_k=10 retrieval."""
        experiment = ABExperiment(
            experiment_id=str(uuid.uuid4()),
            name="retrieval-top-k-comparison",
            variants=[
                ExperimentVariant("control", "top_k=5", 0.5, {"top_k": 5}),
                ExperimentVariant("treatment", "top_k=10", 0.5, {"top_k": 10}),
            ],
            primary_metric="answer_relevancy",
            min_sample_size=200,
        )
        self.experiments[experiment.experiment_id] = experiment
        return experiment

    def create_reranker_experiment(self) -> ABExperiment:
        """Test with vs without reranking."""
        experiment = ABExperiment(
            experiment_id=str(uuid.uuid4()),
            name="reranker-ablation",
            variants=[
                ExperimentVariant("no_rerank", "retrieval only", 0.5, {"use_reranker": False}),
                ExperimentVariant("with_rerank", "retrieval + rerank", 0.5, {"use_reranker": True}),
            ],
            primary_metric="faithfulness",
            min_sample_size=100,
        )
        self.experiments[experiment.experiment_id] = experiment
        return experiment

    def get_all_results(self) -> List[Dict]:
        return [exp.analyze() for exp in self.experiments.values()]
