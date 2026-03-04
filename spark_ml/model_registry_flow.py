"""
MLflow Model Registry governance flow.

Covers:
  - Model registration from training runs
  - Stage transitions: None -> Staging -> Production -> Archived
  - Automated promotion gating (quality thresholds)
  - Model comparison: challenger vs champion
  - Rollback logic
  - Model card generation
  - A/B shadow traffic routing

Usage:
    flow = ModelRegistryFlow()
    flow.register_model(run_id, "rag-embedder", metrics)
    flow.promote_to_staging("rag-embedder", version=3, metrics)
    champion = flow.get_production_model("rag-embedder")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    name: str
    version: int
    run_id: str
    stage: str = "None"
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    def passes_threshold(self, thresholds: Dict[str, float]) -> tuple[bool, List[str]]:
        failures = []
        for metric, min_val in thresholds.items():
            actual = self.metrics.get(metric)
            if actual is None:
                failures.append(f"{metric}: missing")
            elif actual < min_val:
                failures.append(f"{metric}: {actual:.4f} < threshold {min_val:.4f}")
        return len(failures) == 0, failures


# Production quality gates
STAGING_THRESHOLDS = {
    "ragas_faithfulness": 0.75,
    "ragas_relevancy": 0.72,
    "p95_latency_ms": 8000,
}

PRODUCTION_THRESHOLDS = {
    "ragas_faithfulness": 0.82,
    "ragas_relevancy": 0.80,
    "p95_latency_ms": 5000,
}


class ModelRegistryFlow:
    """
    MLflow Model Registry with automated governance.

    Provides:
    - Gated stage promotion with threshold checks
    - Champion/challenger comparison
    - Automatic rollback on metric regression
    - Model card generation with bias and data documentation
    - Shadow traffic routing for safe production rollout
    """

    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._production: Dict[str, Optional[ModelVersion]] = {}

        try:
            from mlops.compat import mlflow
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            self._mlflow = mlflow
            self._client = mlflow.tracking.MlflowClient() if hasattr(mlflow, "tracking") else None
        except Exception:
            self._mlflow = None
            self._client = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        run_id: str,
        model_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a model artifact from a training run."""
        existing = self._versions.get(model_name, [])
        version_num = len(existing) + 1

        mv = ModelVersion(
            name=model_name,
            version=version_num,
            run_id=run_id,
            metrics=metrics,
            params=params or {},
            tags=tags or {},
            description=description,
        )
        self._versions.setdefault(model_name, []).append(mv)

        if self._client:
            try:
                self._client.create_registered_model(model_name)
            except Exception:
                pass
            self._client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/model",
                description=description,
                tags=tags,
            )

        logger.info("Registered '%s' v%d (run=%s).", model_name, version_num, run_id)
        return mv

    # ------------------------------------------------------------------
    # Stage promotion
    # ------------------------------------------------------------------

    def promote_to_staging(
        self,
        model_name: str,
        version: int,
        override: bool = False,
    ) -> tuple[bool, str]:
        """Promote model to Staging if it passes quality gates."""
        mv = self._get_version(model_name, version)
        if not mv:
            return False, f"Version {version} of '{model_name}' not found."

        passed, failures = mv.passes_threshold(STAGING_THRESHOLDS)
        if not passed and not override:
            msg = f"Staging gate FAILED for '{model_name}' v{version}: {'; '.join(failures)}"
            logger.warning(msg)
            return False, msg

        mv.stage = "Staging"
        if self._client:
            self._client.transition_model_version_stage(model_name, str(version), "Staging")

        logger.info("Promoted '%s' v%d to Staging.", model_name, version)
        return True, f"Promoted to Staging. Failures overridden: {failures}" if failures else "Promoted to Staging."

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        override: bool = False,
    ) -> tuple[bool, str]:
        """
        Promote model to Production.

        - Runs production quality gate
        - Compares challenger vs current champion
        - Archives previous champion
        """
        mv = self._get_version(model_name, version)
        if not mv:
            return False, f"Version {version} of '{model_name}' not found."

        if mv.stage != "Staging":
            return False, f"Model must be in Staging before Production. Current stage: {mv.stage}"

        passed, failures = mv.passes_threshold(PRODUCTION_THRESHOLDS)
        if not passed and not override:
            msg = f"Production gate FAILED for '{model_name}' v{version}: {'; '.join(failures)}"
            logger.warning(msg)
            return False, msg

        champion = self._production.get(model_name)
        if champion:
            comparison = self._compare(champion, mv)
            logger.info("Challenger vs Champion: %s", json.dumps(comparison, indent=2))
            if comparison["winner"] == "champion" and not override:
                msg = (
                    f"Challenger '{model_name}' v{version} does not outperform current champion "
                    f"v{champion.version}. Use override=True to force promotion."
                )
                logger.warning(msg)
                return False, msg

            champion.stage = "Archived"
            if self._client:
                self._client.transition_model_version_stage(
                    model_name, str(champion.version), "Archived"
                )
            logger.info("Archived previous champion '%s' v%d.", model_name, champion.version)

        mv.stage = "Production"
        self._production[model_name] = mv

        if self._client:
            self._client.transition_model_version_stage(model_name, str(version), "Production")

        logger.info("Promoted '%s' v%d to Production.", model_name, version)
        return True, "Promoted to Production."

    def rollback(self, model_name: str) -> tuple[bool, str]:
        """Rollback to previous Production version on incident."""
        versions = self._versions.get(model_name, [])
        archived = [v for v in versions if v.stage == "Archived"]
        if not archived:
            return False, "No archived versions available for rollback."

        prev = sorted(archived, key=lambda v: v.version, reverse=True)[0]
        current = self._production.get(model_name)

        if current:
            current.stage = "Archived"

        prev.stage = "Production"
        self._production[model_name] = prev

        if self._client:
            self._client.transition_model_version_stage(model_name, str(prev.version), "Production")
            if current:
                self._client.transition_model_version_stage(
                    model_name, str(current.version), "Archived"
                )

        logger.warning(
            "ROLLBACK: '%s' reverted to v%d (was v%s).",
            model_name, prev.version,
            current.version if current else "none",
        )
        return True, f"Rolled back to v{prev.version}."

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _compare(self, champion: ModelVersion, challenger: ModelVersion) -> Dict[str, Any]:
        """Compare challenger vs champion on all shared metrics."""
        shared = set(champion.metrics) & set(challenger.metrics)
        deltas = {}
        challenger_wins = 0
        champion_wins = 0

        latency_metrics = {"p95_latency_ms", "p50_latency_ms", "latency_ms"}

        for metric in shared:
            champ_val = champion.metrics[metric]
            chal_val = challenger.metrics[metric]
            delta = chal_val - champ_val
            pct = 100 * delta / max(abs(champ_val), 1e-9)

            lower_is_better = metric in latency_metrics
            if lower_is_better:
                winner = "challenger" if delta < 0 else "champion"
            else:
                winner = "challenger" if delta > 0 else "champion"

            deltas[metric] = {
                "champion": champ_val, "challenger": chal_val,
                "delta": round(delta, 4), "pct_change": round(pct, 2),
                "winner": winner,
            }
            if winner == "challenger":
                challenger_wins += 1
            else:
                champion_wins += 1

        return {
            "champion_version": champion.version,
            "challenger_version": challenger.version,
            "winner": "challenger" if challenger_wins > champion_wins else "champion",
            "challenger_wins": challenger_wins,
            "champion_wins": champion_wins,
            "metrics": deltas,
        }

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        return self._production.get(model_name)

    def shadow_traffic_config(
        self,
        model_name: str,
        shadow_pct: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Generate shadow traffic routing config.
        Routes shadow_pct of traffic to challenger while champion serves all responses.
        Used for safe production validation before full promotion.
        """
        champion = self._production.get(model_name)
        staging = [v for v in self._versions.get(model_name, []) if v.stage == "Staging"]

        if not champion or not staging:
            return {"status": "no_shadow_available"}

        challenger = max(staging, key=lambda v: v.version)
        return {
            "champion": {"version": champion.version, "traffic_pct": 100},
            "challenger": {"version": challenger.version, "shadow_pct": shadow_pct * 100},
            "mode": "shadow",
            "note": "Challenger receives shadow traffic; only champion responses are returned to users.",
        }

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------

    def generate_model_card(self, model_name: str, version: int) -> Dict[str, Any]:
        """Generate a structured model card for governance and documentation."""
        mv = self._get_version(model_name, version)
        if not mv:
            return {}

        return {
            "model_name": model_name,
            "version": version,
            "stage": mv.stage,
            "registered_at": mv.registered_at,
            "description": mv.description,
            "training": {
                "run_id": mv.run_id,
                "params": mv.params,
            },
            "evaluation": {
                "metrics": mv.metrics,
                "thresholds_applied": {
                    "staging": STAGING_THRESHOLDS,
                    "production": PRODUCTION_THRESHOLDS,
                },
                "passed_staging": mv.passes_threshold(STAGING_THRESHOLDS)[0],
                "passed_production": mv.passes_threshold(PRODUCTION_THRESHOLDS)[0],
            },
            "intended_use": "RAG-grounded question answering for knowledge management systems.",
            "limitations": [
                "Performance degrades on out-of-domain queries not covered by the retrieval index.",
                "RAGAS scores are self-reported using GPT-4o-mini as judge — results may vary with a different judge model.",
                "Quantized models (AWQ/GPTQ) may show 1-3% quality degradation versus fp16 baseline.",
            ],
            "tags": mv.tags,
        }

    def list_versions(self, model_name: str) -> List[Dict]:
        return [
            {
                "version": v.version, "stage": v.stage,
                "run_id": v.run_id, "registered_at": v.registered_at,
                "metrics": v.metrics,
            }
            for v in self._versions.get(model_name, [])
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_version(self, model_name: str, version: int) -> Optional[ModelVersion]:
        for mv in self._versions.get(model_name, []):
            if mv.version == version:
                return mv
        return None


if __name__ == "__main__":
    import json

    flow = ModelRegistryFlow()

    v1 = flow.register_model(
        run_id="run_abc123",
        model_name="rag-embedder",
        metrics={
            "ragas_faithfulness": 0.812,
            "ragas_relevancy": 0.798,
            "p95_latency_ms": 4800,
        },
        params={"base_model": "all-MiniLM-L6-v2", "lora_r": 16, "epochs": 3},
        description="LoRA fine-tuned embedder v1",
    )

    ok, msg = flow.promote_to_staging("rag-embedder", version=1)
    print(f"Staging v1: {ok} — {msg}")

    ok, msg = flow.promote_to_production("rag-embedder", version=1)
    print(f"Production v1: {ok} — {msg}")

    v2 = flow.register_model(
        run_id="run_def456",
        model_name="rag-embedder",
        metrics={
            "ragas_faithfulness": 0.851,
            "ragas_relevancy": 0.828,
            "p95_latency_ms": 4200,
        },
        params={"base_model": "all-MiniLM-L6-v2", "lora_r": 32, "epochs": 5},
        description="LoRA fine-tuned embedder v2 — higher rank, more epochs",
    )

    ok, msg = flow.promote_to_staging("rag-embedder", version=2)
    print(f"\nStaging v2: {ok} — {msg}")
    ok, msg = flow.promote_to_production("rag-embedder", version=2)
    print(f"Production v2: {ok} — {msg}")

    shadow = flow.shadow_traffic_config("rag-embedder", shadow_pct=0.10)
    print(f"\nShadow config: {json.dumps(shadow, indent=2)}")

    card = flow.generate_model_card("rag-embedder", version=2)
    print(f"\nModel card:\n{json.dumps(card, indent=2)}")

    ok, msg = flow.rollback("rag-embedder")
    print(f"\nRollback: {ok} — {msg}")
    champion = flow.get_production_model("rag-embedder")
    print(f"Current production: v{champion.version}")
