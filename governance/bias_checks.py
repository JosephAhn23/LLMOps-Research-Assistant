"""
Fairness and bias evaluation for ML models.

Metrics:
  - Statistical parity difference: P(Y=1|A=1) - P(Y=1|A=0)
  - Equal opportunity difference: TPR(A=1) - TPR(A=0)
  - Disparate impact ratio: P(Y=1|A=1) / P(Y=1|A=0)
  - Average odds difference: mean(TPR diff, FPR diff)

CI integration:
  The check() method returns a CICheckResult that can be used to
  fail a CI build if fairness regression exceeds threshold.

Reference: Hardt et al., "Equality of Opportunity in Supervised Learning", NeurIPS 2016.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GroupMetrics:
    group: str
    n: int
    positive_rate: float
    tpr: float
    fpr: float
    accuracy: float
    precision: float


@dataclass
class FairnessReport:
    group_metrics: Dict[str, GroupMetrics]
    statistical_parity_diff: float
    equal_opportunity_diff: float
    average_odds_diff: float
    disparate_impact_ratio: float
    worst_group: str
    best_group: str
    passes_threshold: bool
    threshold: float
    n_groups: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistical_parity_diff": self.statistical_parity_diff,
            "equal_opportunity_diff": self.equal_opportunity_diff,
            "average_odds_diff": self.average_odds_diff,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "worst_group": self.worst_group,
            "best_group": self.best_group,
            "passes_threshold": self.passes_threshold,
            "threshold": self.threshold,
            "groups": {
                g: {
                    "n": m.n,
                    "positive_rate": round(m.positive_rate, 4),
                    "tpr": round(m.tpr, 4),
                    "accuracy": round(m.accuracy, 4),
                }
                for g, m in self.group_metrics.items()
            },
        }

    def summary(self) -> str:
        lines = [
            f"Fairness Report ({self.n_groups} groups):",
            f"  Statistical parity diff:  {self.statistical_parity_diff:+.4f}",
            f"  Equal opportunity diff:   {self.equal_opportunity_diff:+.4f}",
            f"  Average odds diff:        {self.average_odds_diff:+.4f}",
            f"  Disparate impact ratio:   {self.disparate_impact_ratio:.4f}",
            f"  Worst group: {self.worst_group}  Best group: {self.best_group}",
            f"  Passes threshold (<{self.threshold}): {'YES' if self.passes_threshold else 'NO'}",
        ]
        for group, m in self.group_metrics.items():
            lines.append(
                f"  [{group}] n={m.n} pos_rate={m.positive_rate:.4f} "
                f"tpr={m.tpr:.4f} acc={m.accuracy:.4f}"
            )
        return "\n".join(lines)


class BiasChecker:
    """
    Evaluate model predictions for fairness across demographic groups.

    Usage:
        checker = BiasChecker(threshold=0.10)
        report = checker.evaluate(predictions, labels, groups)
        if not report.passes_threshold:
            raise ValueError("Fairness regression detected!")
    """

    def __init__(self, threshold: float = 0.10, positive_class: int = 1):
        self.threshold = threshold
        self.positive_class = positive_class

    def evaluate(
        self,
        predictions: List[int],
        labels: List[int],
        groups: List[str],
    ) -> FairnessReport:
        group_names = sorted(set(groups))
        group_metrics: Dict[str, GroupMetrics] = {}

        for group in group_names:
            idx = [i for i, g in enumerate(groups) if g == group]
            preds = [predictions[i] for i in idx]
            labs = [labels[i] for i in idx]
            n = len(preds)
            if n == 0:
                continue

            tp = sum(1 for p, l in zip(preds, labs) if p == self.positive_class and l == self.positive_class)
            tn = sum(1 for p, l in zip(preds, labs) if p != self.positive_class and l != self.positive_class)
            fp = sum(1 for p, l in zip(preds, labs) if p == self.positive_class and l != self.positive_class)
            fn = sum(1 for p, l in zip(preds, labs) if p != self.positive_class and l == self.positive_class)

            n_pos = tp + fn
            n_neg = tn + fp

            group_metrics[group] = GroupMetrics(
                group=group,
                n=n,
                positive_rate=sum(1 for p in preds if p == self.positive_class) / n,
                tpr=tp / max(n_pos, 1),
                fpr=fp / max(n_neg, 1),
                accuracy=(tp + tn) / n,
                precision=tp / max(tp + fp, 1),
            )

        pos_rates = [m.positive_rate for m in group_metrics.values()]
        tprs = [m.tpr for m in group_metrics.values()]
        fprs = [m.fpr for m in group_metrics.values()]
        accuracies = {g: m.accuracy for g, m in group_metrics.items()}

        spd = max(pos_rates) - min(pos_rates) if len(pos_rates) > 1 else 0.0
        eod = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0
        aod = (
            (max(tprs) - min(tprs) + max(fprs) - min(fprs)) / 2
            if len(tprs) > 1 else 0.0
        )
        di = min(pos_rates) / max(max(pos_rates), 1e-9) if pos_rates else 1.0

        worst = min(accuracies, key=lambda g: accuracies[g]) if accuracies else "unknown"
        best = max(accuracies, key=lambda g: accuracies[g]) if accuracies else "unknown"

        passes = spd < self.threshold and eod < self.threshold

        return FairnessReport(
            group_metrics=group_metrics,
            statistical_parity_diff=round(spd, 4),
            equal_opportunity_diff=round(eod, 4),
            average_odds_diff=round(aod, 4),
            disparate_impact_ratio=round(di, 4),
            worst_group=worst,
            best_group=best,
            passes_threshold=passes,
            threshold=self.threshold,
            n_groups=len(group_metrics),
        )

    def compare(
        self,
        baseline_report: FairnessReport,
        current_report: FairnessReport,
        regression_threshold: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Detect fairness regression between baseline and current model.
        Returns (regression_detected, explanation).
        """
        spd_delta = current_report.statistical_parity_diff - baseline_report.statistical_parity_diff
        eod_delta = current_report.equal_opportunity_diff - baseline_report.equal_opportunity_diff

        regressions = []
        if spd_delta > regression_threshold:
            regressions.append(
                f"Statistical parity worsened by {spd_delta:+.4f} "
                f"(baseline={baseline_report.statistical_parity_diff:.4f}, "
                f"current={current_report.statistical_parity_diff:.4f})"
            )
        if eod_delta > regression_threshold:
            regressions.append(
                f"Equal opportunity worsened by {eod_delta:+.4f} "
                f"(baseline={baseline_report.equal_opportunity_diff:.4f}, "
                f"current={current_report.equal_opportunity_diff:.4f})"
            )

        if regressions:
            return True, "Fairness regression detected: " + "; ".join(regressions)
        return False, "No fairness regression detected."
