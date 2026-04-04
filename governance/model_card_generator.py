"""
Model card generator following Mitchell et al. (2019) standard.

Auto-generates from MLflow run metadata + evaluation results.
Produces structured JSON and markdown-formatted model cards.

Standard sections:
  - Model details (name, version, type, license)
  - Intended use and out-of-scope uses
  - Training data (description, preprocessing, PII audit)
  - Evaluation results (metrics, dataset, date)
  - Ethical considerations (fairness, limitations)
  - Recommendations (monitoring, human oversight)

FastAPI endpoint:
  GET /governance/model-card/{model_name}/{version}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    model_name: str
    version: int
    model_type: str
    description: str
    created_at: str
    intended_use: Dict[str, Any]
    training_data: Dict[str, Any]
    evaluation: Dict[str, Any]
    fairness: Dict[str, Any]
    limitations: List[str]
    recommendations: List[str]
    license: str = "MIT"
    contact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.model_name} v{self.version}",
            f"",
            f"> Generated: {self.created_at[:19]} UTC",
            f"",
            f"## Model Details",
            f"",
            f"| Field | Value |",
            f"|:---|:---|",
            f"| **Name** | `{self.model_name}` |",
            f"| **Version** | `{self.version}` |",
            f"| **Type** | {self.model_type} |",
            f"| **License** | {self.license} |",
            f"| **Contact** | {self.contact or 'N/A'} |",
            f"",
            f"**Description:** {self.description}",
            f"",
            f"## Intended Use",
            f"",
            f"**Primary use:** {self.intended_use.get('primary', 'N/A')}",
            f"",
            f"**Out-of-scope uses:**",
        ]
        for oos in self.intended_use.get("out_of_scope", []):
            lines.append(f"- {oos}")

        lines += [
            f"",
            f"## Training Data",
            f"",
            f"| Field | Value |",
            f"|:---|:---|",
        ]
        for k, v in self.training_data.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"| {k} | {v} |")

        lines += [
            f"",
            f"## Evaluation",
            f"",
            f"| Metric | Score |",
            f"|:---|:---|",
        ]
        for k, v in self.evaluation.get("metrics", {}).items():
            lines.append(f"| `{k}` | `{v}` |")

        lines += [
            f"",
            f"**Evaluation dataset:** {self.evaluation.get('dataset', 'N/A')}",
            f"",
            f"## Fairness",
            f"",
        ]
        for k, v in self.fairness.items():
            lines.append(f"**{k}:** {v}")

        lines += [
            f"",
            f"## Limitations",
            f"",
        ]
        for lim in self.limitations:
            lines.append(f"- {lim}")

        lines += [
            f"",
            f"## Recommendations",
            f"",
        ]
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)


class ModelCardGenerator:
    """
    Generates model cards from MLflow metadata + evaluation results.

    In production: pulls run metadata from MLflow tracking server.
    Here: accepts metadata dict directly.
    """

    DEFAULT_LIMITATIONS = [
        "Performance degrades on out-of-domain queries not covered by the retrieval index.",
        "RAGAS scores are self-reported using GPT-4o-mini as judge — results may vary with a different judge model.",
        "Quantized models (AWQ/GPTQ) may show 1-3% quality degradation versus fp16 baseline.",
        "Hallucinations remain possible when retrieved context is incomplete or contradictory.",
        "Language coverage limited to training corpus languages.",
    ]

    DEFAULT_RECOMMENDATIONS = [
        "Run RAGAS evaluation on deployment data monthly.",
        "Monitor demographic parity across user groups quarterly.",
        "Audit PII in any new training data before ingestion.",
        "Maintain human review for high-stakes query categories (medical, legal, financial).",
        "Set up automated alerts for RAGAS score drops > 5% from baseline.",
    ]

    def generate(
        self,
        model_name: str,
        version: int,
        metrics: Dict[str, float],
        model_type: str = "RAG retrieval + generation model",
        description: str = "",
        training_data_description: str = "",
        intended_use: str = "",
        out_of_scope: Optional[List[str]] = None,
        fairness_report: Optional[Dict] = None,
        run_id: Optional[str] = None,
        contact: str = "",
    ) -> ModelCard:
        try:
            mlflow_metadata = self._fetch_mlflow_metadata(run_id) if run_id else {}
        except Exception:
            mlflow_metadata = {}

        merged_metrics = {**mlflow_metadata.get("metrics", {}), **metrics}

        return ModelCard(
            model_name=model_name,
            version=version,
            model_type=model_type,
            description=description or f"LLM-based {model_type} for knowledge management.",
            created_at=datetime.now(timezone.utc).isoformat(),
            intended_use={
                "primary": intended_use or "RAG-grounded question answering for enterprise knowledge management.",
                "out_of_scope": out_of_scope or [
                    "Medical diagnosis or clinical decision support without human oversight.",
                    "Legal advice without qualified legal review.",
                    "High-stakes decisions affecting individual rights without human-in-the-loop.",
                ],
            },
            training_data={
                "description": training_data_description or "Domain-specific text corpus with quality filtering.",
                "preprocessing": ["MinHash deduplication", "quality score filtering (min_words=100)", "language detection"],
                "pii_audit": mlflow_metadata.get("pii_audit", "Not audited"),
                "run_id": run_id or "N/A",
            },
            evaluation={
                "metrics": merged_metrics,
                "dataset": "Held-out test set (10% of corpus, stratified by domain).",
                "date": datetime.now(timezone.utc).isoformat()[:10],
                "judge_model": "GPT-4o-mini",
            },
            fairness=fairness_report or {
                "demographic_parity": "Not evaluated",
                "equalized_odds": "Not evaluated",
                "note": "Run governance.BiasChecker to populate fairness metrics.",
            },
            limitations=self.DEFAULT_LIMITATIONS,
            recommendations=self.DEFAULT_RECOMMENDATIONS,
            contact=contact,
        )

    def _fetch_mlflow_metadata(self, run_id: str) -> Dict[str, Any]:
        try:
            from mlops.compat import mlflow
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            return {
                "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.debug("Could not fetch MLflow metadata for run %s: %s", run_id, e)
            return {}
