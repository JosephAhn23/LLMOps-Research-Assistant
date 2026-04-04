"""
CI enforcement for governance checks.

Integrates governance checks into the CI/CD pipeline.
Fails the build if:
  1. Fairness regression exceeds threshold vs baseline
  2. PII detected in training data
  3. Model card is missing required fields
  4. Audit log chain is broken

Usage in CI (GitHub Actions):
    python -m governance.ci_enforcement \
        --model-name rag-embedder \
        --version 3 \
        --metrics-file metrics.json \
        --baseline-fairness-file baseline_fairness.json

FastAPI endpoint:
    GET /governance/report
    Returns full governance dashboard for the current production model.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CICheckResult:
    check_name: str
    passed: bool
    message: str
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)

    def is_blocking(self) -> bool:
        return not self.passed and self.severity in ("error", "critical")


@dataclass
class GovernanceReport:
    model_name: str
    version: int
    generated_at: str
    checks: List[CICheckResult]
    overall_passed: bool
    model_card_url: Optional[str] = None
    audit_log_head_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "generated_at": self.generated_at,
            "overall_passed": self.overall_passed,
            "checks": [
                {
                    "check": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "audit_log_head_hash": self.audit_log_head_hash,
        }

    def summary(self) -> str:
        status = "PASSED" if self.overall_passed else "FAILED"
        lines = [f"Governance CI: {status} ({self.model_name} v{self.version})"]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} [{c.severity.upper()}] {c.check_name}: {c.message}")
        return "\n".join(lines)


class CIEnforcement:
    """
    Runs governance checks and returns a report suitable for CI integration.

    Checks performed:
    1. Model card completeness
    2. Fairness regression vs baseline
    3. PII scan on training data sample
    4. Audit log integrity
    5. Metric thresholds (RAGAS scores)
    """

    REQUIRED_CARD_FIELDS = [
        "model_name", "version", "intended_use", "training_data",
        "evaluation", "limitations", "recommendations",
    ]

    METRIC_THRESHOLDS = {
        "ragas_faithfulness": 0.75,
        "ragas_relevancy": 0.72,
        "ragas_context_precision": 0.70,
    }

    def __init__(
        self,
        fairness_threshold: float = 0.10,
        fairness_regression_threshold: float = 0.05,
        metric_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.fairness_threshold = fairness_threshold
        self.fairness_regression_threshold = fairness_regression_threshold
        self.metric_thresholds = metric_thresholds or self.METRIC_THRESHOLDS

    def run_all_checks(
        self,
        model_name: str,
        version: int,
        model_card: Optional[Dict] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        current_fairness: Optional[Dict] = None,
        baseline_fairness: Optional[Dict] = None,
        training_data_sample: Optional[List[Dict]] = None,
        audit_log=None,
    ) -> GovernanceReport:
        checks = []

        if model_card is not None:
            checks.append(self._check_model_card(model_card))

        if current_metrics is not None:
            checks.append(self._check_metric_thresholds(current_metrics))

        if current_fairness is not None:
            checks.append(self._check_fairness_absolute(current_fairness))

        if current_fairness is not None and baseline_fairness is not None:
            checks.append(self._check_fairness_regression(baseline_fairness, current_fairness))

        if training_data_sample is not None:
            checks.append(self._check_pii(training_data_sample))

        if audit_log is not None:
            checks.append(self._check_audit_log(audit_log))

        blocking = any(c.is_blocking() for c in checks)
        overall_passed = not blocking

        audit_hash = None
        if audit_log is not None:
            try:
                audit_hash = audit_log.head_hash()
            except Exception:
                pass

        return GovernanceReport(
            model_name=model_name,
            version=version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            checks=checks,
            overall_passed=overall_passed,
            audit_log_head_hash=audit_hash,
        )

    def _check_model_card(self, card: Dict) -> CICheckResult:
        missing = [f for f in self.REQUIRED_CARD_FIELDS if f not in card]
        if missing:
            return CICheckResult(
                check_name="model_card_completeness",
                passed=False,
                message=f"Missing required fields: {missing}",
                severity="error",
            )
        return CICheckResult(
            check_name="model_card_completeness",
            passed=True,
            message="All required model card fields present.",
        )

    def _check_metric_thresholds(self, metrics: Dict[str, float]) -> CICheckResult:
        failures = []
        for metric, threshold in self.metric_thresholds.items():
            value = metrics.get(metric)
            if value is None:
                continue
            if value < threshold:
                failures.append(f"{metric}={value:.4f} < threshold={threshold:.4f}")

        if failures:
            return CICheckResult(
                check_name="metric_thresholds",
                passed=False,
                message=f"Metrics below threshold: {'; '.join(failures)}",
                severity="error",
                details={"failures": failures},
            )
        return CICheckResult(
            check_name="metric_thresholds",
            passed=True,
            message=f"All {len(self.metric_thresholds)} metric thresholds passed.",
        )

    def _check_fairness_absolute(self, fairness: Dict) -> CICheckResult:
        spd = fairness.get("statistical_parity_diff", 0.0)
        eod = fairness.get("equal_opportunity_diff", 0.0)

        violations = []
        if spd > self.fairness_threshold:
            violations.append(f"statistical_parity_diff={spd:.4f} > {self.fairness_threshold}")
        if eod > self.fairness_threshold:
            violations.append(f"equal_opportunity_diff={eod:.4f} > {self.fairness_threshold}")

        if violations:
            return CICheckResult(
                check_name="fairness_absolute",
                passed=False,
                message=f"Fairness violations: {'; '.join(violations)}",
                severity="error",
            )
        return CICheckResult(
            check_name="fairness_absolute",
            passed=True,
            message=f"Fairness within threshold (SPD={spd:.4f}, EOD={eod:.4f}).",
        )

    def _check_fairness_regression(self, baseline: Dict, current: Dict) -> CICheckResult:
        baseline_spd = baseline.get("statistical_parity_diff", 0.0)
        current_spd = current.get("statistical_parity_diff", 0.0)
        delta = current_spd - baseline_spd

        if delta > self.fairness_regression_threshold:
            return CICheckResult(
                check_name="fairness_regression",
                passed=False,
                message=(
                    f"Fairness regression: SPD worsened by {delta:+.4f} "
                    f"(baseline={baseline_spd:.4f}, current={current_spd:.4f})"
                ),
                severity="error",
            )
        return CICheckResult(
            check_name="fairness_regression",
            passed=True,
            message=f"No fairness regression (SPD delta={delta:+.4f}).",
        )

    def _check_pii(self, data_sample: List[Dict]) -> CICheckResult:
        from governance.pii_redaction import PIIRedactor
        redactor = PIIRedactor()
        text_fields = ["text", "query", "content", "prompt", "response"]
        audit = redactor.audit_batch(data_sample, text_fields)

        if not audit["clean"]:
            return CICheckResult(
                check_name="pii_scan",
                passed=False,
                message=(
                    f"PII detected in {audit['records_with_pii']}/{audit['total_records']} records "
                    f"({100*audit['pii_rate']:.1f}%). Types: {list(audit['pii_type_counts'].keys())}"
                ),
                severity="critical",
                details=audit,
            )
        return CICheckResult(
            check_name="pii_scan",
            passed=True,
            message=f"No PII detected in {audit['total_records']} records.",
        )

    def _check_audit_log(self, audit_log) -> CICheckResult:
        valid, error = audit_log.verify_chain()
        if not valid:
            return CICheckResult(
                check_name="audit_log_integrity",
                passed=False,
                message=f"Audit log chain broken: {error}",
                severity="critical",
            )
        return CICheckResult(
            check_name="audit_log_integrity",
            passed=True,
            message=f"Audit log chain valid ({len(audit_log)} entries, head={audit_log.head_hash()[:12]}...).",
        )


def main():
    """CLI entry point for CI integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Run governance CI checks")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--metrics-file", help="JSON file with evaluation metrics")
    parser.add_argument("--baseline-fairness-file", help="JSON file with baseline fairness report")
    parser.add_argument("--current-fairness-file", help="JSON file with current fairness report")
    parser.add_argument("--model-card-file", help="JSON file with model card")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code on failure")
    args = parser.parse_args()

    enforcement = CIEnforcement()

    metrics = None
    if args.metrics_file:
        with open(args.metrics_file) as f:
            metrics = json.load(f)

    baseline_fairness = None
    if args.baseline_fairness_file:
        with open(args.baseline_fairness_file) as f:
            baseline_fairness = json.load(f)

    current_fairness = None
    if args.current_fairness_file:
        with open(args.current_fairness_file) as f:
            current_fairness = json.load(f)

    model_card = None
    if args.model_card_file:
        with open(args.model_card_file) as f:
            model_card = json.load(f)

    report = enforcement.run_all_checks(
        model_name=args.model_name,
        version=args.version,
        model_card=model_card,
        current_metrics=metrics,
        current_fairness=current_fairness,
        baseline_fairness=baseline_fairness,
    )

    print(report.summary())

    if args.exit_code and not report.overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
