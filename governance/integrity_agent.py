"""
Scientific integrity scans for assistant outputs (CI / batch gates).

Flags:
  * Hedging language that weakens verifiability
  * Phrases that mark unverified assumptions
  * Strong factual claims without an adjacent ``[source_N]`` citation

Optional: verify that cited source indices exist and that the cited chunk text
substantiates the sentence (requires chunk list from ingestion / retrieval).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── Hedging / epistemic weakness (often "pleaser" or under-specified) ───────
HEDGING_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bit\s+is\s+possible\s+that\b", "hedging_possible"),
    (r"\bit\s+might\s+be\b", "hedging_might"),
    (r"\bit\s+could\s+be\b", "hedging_could"),
    (r"\bprobably\b", "hedging_probably"),
    (r"\bperhaps\b", "hedging_perhaps"),
    (r"\bwe\s+believe\b", "hedging_we_believe"),
    (r"\bgenerally\s+speaking\b", "hedging_generally"),
    (r"\bwithout\s+loss\s+of\s+generality\b", "hedging_wlog"),  # OK in proofs — warning only
)

ASSUMPTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bunverified\s+assumption\b", "assumption_unverified"),
    (r"\bwe\s+assume\b", "assumption_we_assume"),
    (r"\bfor\s+simplicity\b", "assumption_simplicity"),
    (r"\bfor\s+convenience\b", "assumption_convenience"),
    (r"\bneglecting\b.+\bterms?\b", "assumption_neglect"),
)

# Sentence looks like a numeric / definitional claim
_STRONG_CLAIM_HINT = re.compile(
    r"(?:\b\d+(?:\.\d+)?(?:\s*%|\s*K|\s*MeV|\s*GeV)?\b)|"
    r"\b(?:equals|proves|implies|therefore|thus)\b|"
    r"\btheorem\b|\blemma\b|\bcorollary\b",
    re.IGNORECASE,
)

_CITATION = re.compile(r"\[source_(\d+)\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class IntegrityViolation:
    rule_id: str
    severity: str  # "error" | "warning"
    message: str
    excerpt: str = ""


@dataclass
class IntegrityReport:
    passed: bool
    violations: List[IntegrityViolation] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def blocking(self) -> List[IntegrityViolation]:
        return [v for v in self.violations if v.severity == "error"]


class IntegrityAgent:
    """
    Rule-based integrity scanner. LLM-free so CI stays deterministic.

    Use :meth:`scan` for a full report; :meth:`passes_gate` for boolean + exit code.
    """

    def __init__(
        self,
        *,
        fail_on_hedging: bool = False,
        fail_on_assumption_phrases: bool = True,
        fail_on_uncited_strong_claims: bool = True,
        wlog_as_warning_only: bool = True,
    ) -> None:
        self.fail_on_hedging = fail_on_hedging
        self.fail_on_assumption_phrases = fail_on_assumption_phrases
        self.fail_on_uncited_strong_claims = fail_on_uncited_strong_claims
        self.wlog_as_warning_only = wlog_as_warning_only

    def scan(
        self,
        text: str,
        *,
        chunks: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> IntegrityReport:
        violations: List[IntegrityViolation] = []
        t = text.strip()
        if not t:
            return IntegrityReport(passed=True, violations=[], stats={"empty": True})

        lower = t.lower()
        for pattern, rid in HEDGING_PATTERNS:
            if self.wlog_as_warning_only and rid == "hedging_wlog":
                sev = "warning"
            else:
                sev = "error" if self.fail_on_hedging else "warning"
            m = re.search(pattern, lower, re.IGNORECASE)
            if m:
                violations.append(
                    IntegrityViolation(
                        rule_id=rid,
                        severity=sev,
                        message="Hedging or weak epistemic framing detected.",
                        excerpt=_clip(t[m.start() : m.end() + 40]),
                    )
                )

        for pattern, rid in ASSUMPTION_PATTERNS:
            m = re.search(pattern, lower, re.IGNORECASE)
            if m:
                sev = "error" if self.fail_on_assumption_phrases else "warning"
                violations.append(
                    IntegrityViolation(
                        rule_id=rid,
                        severity=sev,
                        message="Unverified-assumption style phrasing detected.",
                        excerpt=_clip(t[m.start() : m.end() + 40]),
                    )
                )

        if self.fail_on_uncited_strong_claims:
            violations.extend(_scan_uncited_strong_claims(t))

        if chunks is not None:
            violations.extend(_verify_citations_against_chunks(t, chunks))

        blocking = [v for v in violations if v.severity == "error"]
        return IntegrityReport(
            passed=len(blocking) == 0,
            violations=violations,
            stats={
                "n_violations": len(violations),
                "n_errors": len(blocking),
            },
        )

    def passes_gate(self, text: str, **kwargs: Any) -> Tuple[bool, IntegrityReport]:
        report = self.scan(text, **kwargs)
        return report.passed, report


def _clip(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _scan_uncited_strong_claims(text: str) -> List[IntegrityViolation]:
    out: List[IntegrityViolation] = []
    for sent in _SENTENCE_SPLIT.split(text):
        s = sent.strip()
        if len(s) < 20:
            continue
        if not _STRONG_CLAIM_HINT.search(s):
            continue
        if _CITATION.search(s):
            continue
        out.append(
            IntegrityViolation(
                rule_id="uncited_strong_claim",
                severity="error",
                message="Strong factual or formal claim without [source_N] citation in the same sentence.",
                excerpt=_clip(s, 200),
            )
        )
    return out


def _verify_citations_against_chunks(
    text: str,
    chunks: Sequence[Dict[str, Any]],
) -> List[IntegrityViolation]:
    """Ensure each [source_k] index exists; optionally check light overlap with chunk text."""
    violations: List[IntegrityViolation] = []
    for m in _CITATION.finditer(text):
        idx = int(m.group(1))
        if idx < 1 or idx > len(chunks):
            violations.append(
                IntegrityViolation(
                    rule_id="bad_source_index",
                    severity="error",
                    message=f"Citation [source_{idx}] has no matching chunk (index out of range).",
                    excerpt=m.group(0),
                )
            )
            continue
        chunk_text = (chunks[idx - 1].get("text") or "").lower()
        if not chunk_text:
            violations.append(
                IntegrityViolation(
                    rule_id="empty_cited_chunk",
                    severity="error",
                    message=f"Chunk for [source_{idx}] is empty.",
                    excerpt=m.group(0),
                )
            )
            continue
        # Sentence containing citation — require at least one non-trivial token overlap
        start = max(0, text.rfind(".", 0, m.start()) + 1)
        end = text.find(".", m.end())
        if end < 0:
            end = len(text)
        window = text[start:end].lower()
        toks = [w for w in re.findall(r"[a-zA-Z]{4,}", window) if w not in _STOPWORDS]
        hits = sum(1 for w in toks[:12] if w in chunk_text)
        if len(toks) >= 3 and hits == 0:
            violations.append(
                IntegrityViolation(
                    rule_id="citation_not_substantiated",
                    severity="error",
                    message=f"Sentence citing [source_{idx}] has no clear lexical overlap with chunk text.",
                    excerpt=_clip(window, 160),
                )
            )
    return violations


_STOPWORDS = frozenset(
    "that this with from they have been were will would could should source "
    "which their there these those when where while".split()
)
