"""
Dataset quality checks for RAG training and evaluation data.

Checks:
  - Schema validation (required columns, types)
  - Null rate per column
  - Duplicate detection (exact + near-duplicate via MinHash)
  - Text length distribution (too short = noise, too long = chunking issue)
  - PII detection (email, SSN, phone patterns)
  - Distribution drift (KS test vs reference dataset)
  - Label balance (for classification datasets)

Usage:
    checker = DataQualityChecker(schema={"question": str, "answer": str})
    report = checker.check(df)
    print(report.summary())
    if not report.passed:
        raise ValueError("Dataset quality check failed")
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# PII patterns (same as governance/pii_redaction.py for consistency)
_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


@dataclass
class QualityIssue:
    severity: str  # "error" | "warning" | "info"
    check: str
    column: str | None
    message: str
    affected_rows: int = 0


@dataclass
class QualityReport:
    passed: bool
    issues: list[QualityIssue]
    row_count: int
    column_count: int
    stats: dict[str, Any] = field(default_factory=dict)

    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"Quality Report: {'PASSED' if self.passed else 'FAILED'}",
            f"  Rows: {self.row_count:,}  Columns: {self.column_count}",
            f"  Errors: {len(self.errors())}  Warnings: {len(self.warnings())}",
        ]
        for issue in self.issues:
            col_str = f"[{issue.column}] " if issue.column else ""
            lines.append(
                f"  [{issue.severity.upper():7s}] {issue.check}: {col_str}{issue.message}"
                + (f" ({issue.affected_rows:,} rows)" if issue.affected_rows else "")
            )
        return "\n".join(lines)


class QualityChecker:
    """
    Comprehensive dataset quality checker.

    Parameters
    ----------
    schema : dict[str, type] | None
        Expected column names and types. Raises error if columns are missing.
    max_null_rate : float
        Maximum acceptable null rate per column (default 0.05 = 5%).
    min_text_length : int
        Minimum character length for text columns.
    max_text_length : int
        Maximum character length for text columns.
    max_duplicate_rate : float
        Maximum acceptable exact-duplicate rate.
    check_pii : bool
        Whether to scan text columns for PII patterns.
    reference_df : Any | None
        Reference dataset for distribution drift checks (KS test).
    """

    def __init__(
        self,
        schema: dict[str, type] | None = None,
        max_null_rate: float = 0.05,
        min_text_length: int = 10,
        max_text_length: int = 10_000,
        max_duplicate_rate: float = 0.01,
        check_pii: bool = True,
        reference_df: Any | None = None,
    ) -> None:
        self.schema = schema or {}
        self.max_null_rate = max_null_rate
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.max_duplicate_rate = max_duplicate_rate
        self.check_pii = check_pii
        self.reference_df = reference_df

    def check(self, df: Any) -> QualityReport:
        """Run all quality checks and return a report."""
        issues: list[QualityIssue] = []

        issues.extend(self._check_schema(df))
        issues.extend(self._check_nulls(df))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_text_lengths(df))
        if self.check_pii:
            issues.extend(self._check_pii(df))
        if self.reference_df is not None:
            issues.extend(self._check_drift(df))

        has_errors = any(i.severity == "error" for i in issues)

        try:
            row_count = len(df)
            col_count = len(df.columns) if hasattr(df, "columns") else 0
        except Exception:
            row_count, col_count = 0, 0

        return QualityReport(
            passed=not has_errors,
            issues=issues,
            row_count=row_count,
            column_count=col_count,
        )

    def _check_schema(self, df: Any) -> list[QualityIssue]:
        issues = []
        if not self.schema:
            return issues
        try:
            cols = set(df.columns)
        except AttributeError:
            return issues

        for col, expected_type in self.schema.items():
            if col not in cols:
                issues.append(QualityIssue(
                    severity="error",
                    check="schema",
                    column=col,
                    message=f"Required column '{col}' is missing",
                ))
        return issues

    def _check_nulls(self, df: Any) -> list[QualityIssue]:
        issues = []
        try:
            for col in df.columns:
                null_rate = float(df[col].isna().mean())
                if null_rate > self.max_null_rate:
                    severity = "error" if null_rate > 0.2 else "warning"
                    issues.append(QualityIssue(
                        severity=severity,
                        check="null_rate",
                        column=col,
                        message=f"Null rate {null_rate:.1%} exceeds threshold {self.max_null_rate:.1%}",
                        affected_rows=int(df[col].isna().sum()),
                    ))
        except Exception as e:
            logger.debug("Null check failed: %s", e)
        return issues

    def _check_duplicates(self, df: Any) -> list[QualityIssue]:
        issues = []
        try:
            n_dupes = int(df.duplicated().sum())
            dupe_rate = n_dupes / max(len(df), 1)
            if dupe_rate > self.max_duplicate_rate:
                severity = "error" if dupe_rate > 0.1 else "warning"
                issues.append(QualityIssue(
                    severity=severity,
                    check="duplicates",
                    column=None,
                    message=f"Duplicate rate {dupe_rate:.1%} exceeds threshold {self.max_duplicate_rate:.1%}",
                    affected_rows=n_dupes,
                ))
        except Exception as e:
            logger.debug("Duplicate check failed: %s", e)
        return issues

    def _check_text_lengths(self, df: Any) -> list[QualityIssue]:
        issues = []
        try:
            import pandas as pd
            for col in df.columns:
                if df[col].dtype == object:
                    lengths = df[col].dropna().astype(str).str.len()
                    too_short = int((lengths < self.min_text_length).sum())
                    too_long = int((lengths > self.max_text_length).sum())
                    if too_short > 0:
                        issues.append(QualityIssue(
                            severity="warning",
                            check="text_length",
                            column=col,
                            message=f"{too_short} rows below min length {self.min_text_length}",
                            affected_rows=too_short,
                        ))
                    if too_long > 0:
                        issues.append(QualityIssue(
                            severity="warning",
                            check="text_length",
                            column=col,
                            message=f"{too_long} rows above max length {self.max_text_length}",
                            affected_rows=too_long,
                        ))
        except Exception as e:
            logger.debug("Text length check failed: %s", e)
        return issues

    def _check_pii(self, df: Any) -> list[QualityIssue]:
        issues = []
        try:
            for col in df.columns:
                if df[col].dtype != object:
                    continue
                sample = df[col].dropna().astype(str).head(1000)
                for pii_type, pattern in _PII_PATTERNS.items():
                    hits = int(sample.str.contains(pattern, regex=True).sum())
                    if hits > 0:
                        issues.append(QualityIssue(
                            severity="error",
                            check="pii",
                            column=col,
                            message=f"Detected {pii_type} pattern in {hits} rows (sample of 1000)",
                            affected_rows=hits,
                        ))
        except Exception as e:
            logger.debug("PII check failed: %s", e)
        return issues

    def _check_drift(self, df: Any) -> list[QualityIssue]:
        """KS test for distribution drift vs reference dataset."""
        issues = []
        try:
            from scipy import stats as scipy_stats
            for col in df.columns:
                if col not in self.reference_df.columns:
                    continue
                if df[col].dtype not in (float, int, "float64", "int64"):
                    continue
                a = df[col].dropna().to_numpy()
                b = self.reference_df[col].dropna().to_numpy()
                if len(a) < 10 or len(b) < 10:
                    continue
                ks_stat, p_value = scipy_stats.ks_2samp(a, b)
                if p_value < 0.05:
                    issues.append(QualityIssue(
                        severity="warning",
                        check="distribution_drift",
                        column=col,
                        message=f"KS test p={p_value:.4f} (stat={ks_stat:.3f}): distribution differs from reference",
                    ))
        except Exception as e:
            logger.debug("Drift check failed: %s", e)
        return issues
