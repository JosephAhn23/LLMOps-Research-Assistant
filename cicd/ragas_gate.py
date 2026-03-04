"""
RAGAS Regression Gate — CI/CD quality gate script.

Compares current RAGAS scores against a saved baseline and blocks deploys
when any metric regresses beyond the configured tolerance.

Usage:
    python cicd/ragas_gate.py \\
        --scores mlops/ragas_scores.json \\
        --baseline mlops/ragas_baseline.json \\
        [--tolerance 0.03] \\
        [--thresholds path/to/thresholds.json] \\
        [--output-file gate_result.json]

Exit codes:
    0  — all metrics pass (deploy allowed)
    1  — threshold violation or regression detected (deploy blocked)

GitHub Actions output variables written to $GITHUB_OUTPUT (when set):
    passed=true|false
    faithfulness=0.8500
    answer_relevancy=0.8300
    context_precision=0.7900
    context_recall=0.8100
    (one output per metric in the scores file)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds — absolute minimums that must always be met
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.78,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

# How much a score can drop vs. baseline before triggering failure
DEFAULT_REGRESSION_TOLERANCE: float = 0.03


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

class RAGASGate:
    """
    Evaluates RAGAS scores against absolute thresholds and a saved baseline.

    Attributes:
        thresholds:  Per-metric minimum acceptable scores (hard floor).
        tolerance:   Maximum allowed absolute drop vs baseline before failure.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.tolerance = tolerance

    def check_thresholds(
        self, scores: Dict[str, float]
    ) -> Tuple[bool, Dict[str, dict]]:
        """Return (passed, failures) where failures maps metric → {score, threshold}."""
        failures = {}
        for metric, threshold in self.thresholds.items():
            score = scores.get(metric)
            if score is None:
                logger.warning("Metric '%s' missing from scores — skipping", metric)
                continue
            if score < threshold:
                logger.error("FAIL | %s: %.4f < threshold %.4f", metric, score, threshold)
                failures[metric] = {"score": round(score, 4), "threshold": threshold}
            else:
                logger.info("PASS | %s: %.4f >= threshold %.4f", metric, score, threshold)
        return len(failures) == 0, failures

    def check_regression(
        self, scores: Dict[str, float], baseline: Dict[str, float]
    ) -> Tuple[bool, Dict[str, dict]]:
        """
        Return (passed, regressions) where regressions maps metric →
        {baseline, current, drop}.
        """
        regressions = {}
        for metric, current in scores.items():
            base = baseline.get(metric)
            if base is None:
                continue
            drop = base - current
            if drop > self.tolerance:
                logger.error(
                    "REGRESSION | %s: dropped %.4f (%0.4f → %.4f), tolerance=%.2f",
                    metric, drop, base, current, self.tolerance,
                )
                regressions[metric] = {
                    "baseline": round(base, 4),
                    "current": round(current, 4),
                    "drop": round(drop, 4),
                }
        return len(regressions) == 0, regressions

    def evaluate(
        self,
        scores: Dict[str, float],
        baseline: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Run all gate checks and return a structured result dict.

        Returns:
            {
                "passed": bool,
                "threshold_failures": {...},
                "regressions": {...},
                "scores": {...},
            }
        """
        result: dict = {
            "passed": True,
            "threshold_failures": {},
            "regressions": {},
            "scores": scores,
        }

        thresh_ok, thresh_failures = self.check_thresholds(scores)
        if not thresh_ok:
            result["passed"] = False
            result["threshold_failures"] = thresh_failures

        if baseline:
            reg_ok, regressions = self.check_regression(scores, baseline)
            if not reg_ok:
                result["passed"] = False
                result["regressions"] = regressions
        else:
            logger.warning("No baseline provided — skipping regression comparison")

        return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    scores = result["scores"]
    print("\n" + "=" * 60)
    print("  RAGAS REGRESSION GATE REPORT")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Score':>8}")
    print("-" * 35)
    for metric, score in scores.items():
        flag = ""
        if metric in result["threshold_failures"]:
            flag = f"  ✗ BELOW THRESHOLD ({result['threshold_failures'][metric]['threshold']:.2f})"
        elif metric in result["regressions"]:
            drop = result["regressions"][metric]["drop"]
            flag = f"  ✗ REGRESSION (-{drop:.4f})"
        print(f"  {metric:<23} {score:>8.4f}{flag}")

    if result["threshold_failures"]:
        print("\n[THRESHOLD FAILURES]")
        for m, info in result["threshold_failures"].items():
            print(f"  {m}: {info['score']:.4f} < {info['threshold']:.4f}")

    if result["regressions"]:
        print("\n[REGRESSIONS vs BASELINE]")
        for m, info in result["regressions"].items():
            print(f"  {m}: {info['baseline']:.4f} → {info['current']:.4f} (drop={info['drop']:.4f})")

    status = "PASSED ✓" if result["passed"] else "FAILED ✗"
    print(f"\nGate status: {status}")
    print("=" * 60 + "\n")


def write_github_outputs(result: dict) -> None:
    """
    Write output variables to $GITHUB_OUTPUT for downstream job conditions.

    Writes:
      passed=true|false
      <metric>=<score>  (one line per metric in the scores dict)
    """
    gho = os.environ.get("GITHUB_OUTPUT")
    if not gho:
        return
    with open(gho, "a") as f:
        f.write(f"passed={'true' if result['passed'] else 'false'}\n")
        for metric, score in result["scores"].items():
            f.write(f"{metric}={score:.4f}\n")
    logger.info("GitHub Actions outputs written to %s", gho)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_scores(path: str) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        logger.error("Scores file not found: %s", path)
        sys.exit(1)
    data = json.loads(p.read_text())
    # Support both flat {"faithfulness": 0.85} and nested {"scores": {...}}
    return data.get("scores", data)


def load_baseline(path: str) -> Optional[Dict[str, float]]:
    p = Path(path)
    if not p.exists():
        logger.warning("Baseline file not found: %s — skipping regression check", path)
        return None
    data = json.loads(p.read_text())
    return data.get("scores", data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAGAS regression gate — blocks deploys on quality regression."
    )
    p.add_argument(
        "--scores",
        required=True,
        help="Path to current RAGAS scores JSON",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline RAGAS scores JSON",
    )
    p.add_argument(
        "--thresholds",
        default=None,
        help="Path to JSON file with per-metric minimum thresholds",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_REGRESSION_TOLERANCE,
        help=f"Max allowed absolute drop vs baseline (default: {DEFAULT_REGRESSION_TOLERANCE})",
    )
    p.add_argument(
        "--output-file",
        default=None,
        help="Optional path to write gate result JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scores = load_scores(args.scores)
    baseline = load_baseline(args.baseline) if args.baseline else None
    thresholds = (
        json.loads(Path(args.thresholds).read_text())
        if args.thresholds and Path(args.thresholds).exists()
        else DEFAULT_THRESHOLDS
    )

    gate = RAGASGate(thresholds=thresholds, tolerance=args.tolerance)
    result = gate.evaluate(scores, baseline)

    print_report(result)
    write_github_outputs(result)

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(result, indent=2))
        logger.info("Gate result written to %s", args.output_file)

    if result["passed"]:
        logger.info("✅ Quality gate PASSED")
        sys.exit(0)
    else:
        logger.error("❌ Quality gate FAILED — blocking deploy")
        sys.exit(1)


if __name__ == "__main__":
    main()
