"""
Data quality assessment + deduplication pipeline.
Covers: Data quality assessment, Pandas, data filtering
"""
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class QualityMetrics:
    avg_word_length: float
    punctuation_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    duplicate_line_ratio: float
    passed: bool


class DataQualityFilter:
    """
    Rule-based quality filter matching production data pipeline standards.
    Covers: Data quality assessment techniques
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100_000,
        min_avg_word_len: float = 3.0,
        max_digit_ratio: float = 0.2,
        max_uppercase_ratio: float = 0.3,
        max_duplicate_line_ratio: float = 0.3,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_len = min_avg_word_len
        self.max_digit_ratio = max_digit_ratio
        self.max_uppercase_ratio = max_uppercase_ratio
        self.max_duplicate_line_ratio = max_duplicate_line_ratio

    def assess(self, text: str) -> QualityMetrics:
        words = text.split()
        n_words = len(words)

        if n_words == 0:
            return QualityMetrics(0, 0, 0, 0, 0, False)

        avg_word_len = sum(len(w) for w in words) / n_words
        chars = list(text)
        punct_ratio = sum(1 for c in chars if c in ".,;:!?\"'") / max(len(chars), 1)
        digit_ratio = sum(1 for c in chars if c.isdigit()) / max(len(chars), 1)
        upper_ratio = sum(1 for c in chars if c.isupper()) / max(len(chars), 1)

        lines = text.splitlines()
        n_lines = len(lines)
        n_dupes = n_lines - len(set(lines))
        dup_ratio = n_dupes / max(n_lines, 1)

        passed = (
            self.min_words <= n_words <= self.max_words
            and avg_word_len >= self.min_avg_word_len
            and digit_ratio <= self.max_digit_ratio
            and upper_ratio <= self.max_uppercase_ratio
            and dup_ratio <= self.max_duplicate_line_ratio
        )

        return QualityMetrics(avg_word_len, punct_ratio, digit_ratio, upper_ratio, dup_ratio, passed)

    def is_quality(self, text: str) -> bool:
        return self.assess(text).passed

    def assess_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Vectorized quality assessment over a DataFrame."""
        metrics = df[text_col].apply(lambda t: self.assess(t))
        df["avg_word_length"] = metrics.apply(lambda m: m.avg_word_length)
        df["digit_ratio"] = metrics.apply(lambda m: m.digit_ratio)
        df["uppercase_ratio"] = metrics.apply(lambda m: m.uppercase_ratio)
        df["duplicate_line_ratio"] = metrics.apply(lambda m: m.duplicate_line_ratio)
        df["quality_passed"] = metrics.apply(lambda m: m.passed)
        return df


class Deduplicator:
    """
    MinHash-inspired exact + near-duplicate detection.
    Covers: Data quality assessment, deduplication
    """

    def __init__(self, ngram_size: int = 5):
        self.ngram_size = ngram_size
        self.seen_hashes: set = set()

    def hash(self, text: str) -> str:
        """Exact content hash."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def ngram_hash(self, text: str) -> str:
        """N-gram fingerprint for near-duplicate detection."""
        words = text.lower().split()
        ngrams = [" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]
        fingerprint = "|".join(sorted(set(ngrams[:50])))
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def deduplicate_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> Tuple[pd.DataFrame, Dict]:
        """Remove exact + near-duplicates from DataFrame."""
        original_len = len(df)

        df["_exact_hash"] = df[text_col].apply(self.hash)
        df = df.drop_duplicates(subset="_exact_hash")

        df["_ngram_hash"] = df[text_col].apply(self.ngram_hash)
        df = df.drop_duplicates(subset="_ngram_hash")

        df = df.drop(columns=["_exact_hash", "_ngram_hash"])

        stats = {
            "original": original_len,
            "after_dedup": len(df),
            "removed": original_len - len(df),
            "dedup_rate": (original_len - len(df)) / max(original_len, 1),
        }
        return df, stats
