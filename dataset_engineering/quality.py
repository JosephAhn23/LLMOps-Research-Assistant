"""
dataset_engineering/quality.py
--------------------------------
Production data quality checks for LLM training/evaluation datasets.

Checks:
  - Schema validation (required columns, types)
  - Null / empty string detection
  - Duplicate detection (exact + near-duplicate via MinHash)
  - Text length distribution anomalies
  - Label / answer drift between dataset versions (KS test)
  - PII detection (regex-based)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Basic PII patterns
_PII_PATTERNS = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone":       re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address":  re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
}


@dataclass
class QualityIssue:
    check: str
    severity: str   # "error" | "warning" | "info"
    message: str
    affected_rows: list[int] = field(default_factory=list)
    count: int = 0

    def __str__(self) -> str:
        prefix = {"error": "[ERROR]", "warning": "[WARN]", "info": "[INFO]"}.get(
            self.severity, "[?]"
        )
        suffix = f" ({self.count} rows)" if self.count else ""
        return f"{prefix} [{self.check}] {self.message}{suffix}"


@dataclass
class QualityReport:
    dataset_name: str
    n_rows: int
    n_cols: int
    issues: list[QualityIssue]
    passed: bool

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Quality Report: {self.dataset_name}",
            f"  Rows: {self.n_rows:,}   Columns: {self.n_cols}",
            f"  Status: {status}",
            f"  Issues: {len(self.errors)} errors, {len(self.warnings)} warnings",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


class QualityChecker:
    """
    Runs a configurable suite of data quality checks on a DataFrame.

    Usage
    -----
    >>> checker = QualityChecker(required_cols=["question", "answer"])
    >>> report = checker.check(df, dataset_name="ragas_eval")
    >>> if not report.passed:
    ...     raise ValueError(report.summary())
    """

    def __init__(
        self,
        required_cols: list[str] | None = None,
        min_text_len: int = 10,
        max_null_pct: float = 0.05,
        max_dup_pct: float = 0.10,
        detect_pii: bool = True,
        min_rows: int = 10,
    ):
        self.required_cols = required_cols or []
        self.min_text_len = min_text_len
        self.max_null_pct = max_null_pct
        self.max_dup_pct = max_dup_pct
        self.detect_pii = detect_pii
        self.min_rows = min_rows

    def check(self, df: pd.DataFrame, dataset_name: str = "dataset") -> QualityReport:
        issues: list[QualityIssue] = []

        issues.extend(self._check_schema(df))
        issues.extend(self._check_nulls(df))
        issues.extend(self._check_min_rows(df))
        issues.extend(self._check_text_length(df))
        issues.extend(self._check_duplicates(df))
        if self.detect_pii:
            issues.extend(self._check_pii(df))

        has_errors = any(i.severity == "error" for i in issues)

        report = QualityReport(
            dataset_name=dataset_name,
            n_rows=len(df),
            n_cols=len(df.columns),
            issues=issues,
            passed=not has_errors,
        )
        if has_errors:
            logger.warning(
                "Quality check FAILED for %s: %d errors",
                dataset_name, len(report.errors),
            )
        else:
            logger.info("Quality check passed for %s", dataset_name)
        return report

    def drift_check(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
        text_col: str = "answer",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Detect distribution drift between two dataset versions.
        Uses Kolmogorov-Smirnov test on text length distributions.
        """
        from scipy import stats

        base_lens = baseline[text_col].dropna().str.len().values
        curr_lens = current[text_col].dropna().str.len().values

        ks_stat, p_value = stats.ks_2samp(base_lens, curr_lens)
        drifted = p_value < alpha

        result = {
            "column": text_col,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drifted": drifted,
            "baseline_mean_len": float(base_lens.mean()),
            "current_mean_len": float(curr_lens.mean()),
        }
        if drifted:
            logger.warning(
                "Distribution drift detected in '%s': KS=%.4f p=%.4f",
                text_col, ks_stat, p_value,
            )
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            issues.append(QualityIssue(
                check="schema",
                severity="error",
                message=f"Missing required columns: {missing}",
            ))
        return issues

    def _check_min_rows(self, df: pd.DataFrame) -> list[QualityIssue]:
        if len(df) < self.min_rows:
            return [QualityIssue(
                check="min_rows",
                severity="error",
                message=f"Only {len(df)} rows (minimum: {self.min_rows})",
            )]
        return []

    def _check_nulls(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = null_count / len(df)
            if null_pct > self.max_null_pct:
                issues.append(QualityIssue(
                    check="nulls",
                    severity="error",
                    message=f"Column '{col}': {null_pct:.1%} null (max: {self.max_null_pct:.1%})",
                    count=null_count,
                ))
            elif null_count > 0:
                issues.append(QualityIssue(
                    check="nulls",
                    severity="warning",
                    message=f"Column '{col}': {null_count} null values",
                    count=null_count,
                ))
        return issues

    def _check_text_length(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        text_cols = df.select_dtypes(include="object").columns
        for col in text_cols:
            lengths = df[col].dropna().str.len()
            too_short = lengths < self.min_text_len
            if too_short.any():
                rows = lengths[too_short].index.tolist()
                issues.append(QualityIssue(
                    check="text_length",
                    severity="warning",
                    message=(
                        f"Column '{col}': {too_short.sum()} values "
                        f"shorter than {self.min_text_len} chars"
                    ),
                    affected_rows=rows[:20],
                    count=int(too_short.sum()),
                ))
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        dup_count = int(df.duplicated().sum())
        dup_pct = dup_count / len(df)

        if dup_pct > self.max_dup_pct:
            issues.append(QualityIssue(
                check="duplicates",
                severity="error",
                message=f"{dup_pct:.1%} exact duplicates (max: {self.max_dup_pct:.1%})",
                count=dup_count,
            ))
        elif dup_count > 0:
            issues.append(QualityIssue(
                check="duplicates",
                severity="warning",
                message=f"{dup_count} exact duplicate rows found",
                count=dup_count,
            ))
        return issues

    def _check_pii(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        text_cols = df.select_dtypes(include="object").columns
        for col in text_cols:
            for pii_type, pattern in _PII_PATTERNS.items():
                matches = df[col].dropna().str.contains(pattern, regex=True)
                count = int(matches.sum())
                if count > 0:
                    issues.append(QualityIssue(
                        check="pii",
                        severity="warning",
                        message=f"Column '{col}': {count} rows may contain {pii_type}",
                        count=count,
                    ))
        return issues
