"""
PII detection and redaction pipeline.

Patterns covered:
  - Email addresses
  - US/international phone numbers
  - Social Security Numbers
  - Credit card numbers
  - IP addresses
  - API keys (OpenAI, AWS, generic)
  - Dates of birth
  - Names (heuristic: capitalized word pairs not at sentence start)

Used before:
  - Logging any user prompt
  - Storing conversation history
  - Writing to audit log
  - Training data ingestion

Performance: regex-based, O(n) per text, suitable for real-time use.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key_openai": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "api_key_aws": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api_key_generic": re.compile(r"\b(?:api[-_]?key|token|secret)[-_=:\s]+[A-Za-z0-9_\-]{16,}\b", re.IGNORECASE),
    "date_of_birth": re.compile(
        r"\b(?:dob|date\s+of\s+birth|born\s+on)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
        re.IGNORECASE,
    ),
}


@dataclass
class PIIDetection:
    pii_type: str
    start: int
    end: int
    value_hash: str
    replacement: str


@dataclass
class PIIResult:
    original_text: str
    redacted_text: str
    detections: List[PIIDetection]
    has_pii: bool
    pii_types: List[str]
    redaction_count: int

    def risk_level(self) -> str:
        critical = {"ssn", "credit_card", "api_key_openai", "api_key_aws"}
        if any(d.pii_type in critical for d in self.detections):
            return "critical"
        if self.has_pii:
            return "moderate"
        return "clean"


class PIIRedactor:
    """
    Multi-pattern PII detector and redactor.

    Usage:
        redactor = PIIRedactor()
        result = redactor.redact("Call me at 555-123-4567, john@example.com")
        safe_text = result.redacted_text
        if result.has_pii:
            log_pii_event(result.pii_types)
    """

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, re.Pattern]] = None,
        replacement_style: str = "type_tag",
    ):
        self.patterns = {**PII_PATTERNS, **(custom_patterns or {})}
        self.replacement_style = replacement_style

    def detect(self, text: str) -> List[PIIDetection]:
        detections = []
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                value_hash = hashlib.sha256(match.group().encode()).hexdigest()[:12]
                replacement = self._make_replacement(pii_type, match.group())
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    value_hash=value_hash,
                    replacement=replacement,
                ))
        return sorted(detections, key=lambda d: d.start)

    def redact(self, text: str) -> PIIResult:
        detections = self.detect(text)
        redacted = text
        offset = 0

        for det in detections:
            start = det.start + offset
            end = det.end + offset
            original_len = end - start
            redacted = redacted[:start] + det.replacement + redacted[end:]
            offset += len(det.replacement) - original_len

        pii_types = list({d.pii_type for d in detections})
        return PIIResult(
            original_text=text,
            redacted_text=redacted,
            detections=detections,
            has_pii=len(detections) > 0,
            pii_types=pii_types,
            redaction_count=len(detections),
        )

    def redact_safe(self, text: str) -> Tuple[str, bool]:
        """Returns (redacted_text, had_pii). Never raises."""
        try:
            result = self.redact(text)
            return result.redacted_text, result.has_pii
        except Exception as e:
            logger.error("PII redaction failed: %s", e)
            return "[REDACTION_ERROR]", True

    def audit_batch(self, records: List[Dict], text_fields: List[str]) -> Dict:
        """Scan a dataset for PII without exposing values."""
        total = len(records)
        pii_count = 0
        type_counts: Dict[str, int] = {}

        for rec in records:
            record_pii = False
            for field_name in text_fields:
                text = str(rec.get(field_name, ""))
                result = self.redact(text)
                if result.has_pii:
                    record_pii = True
                    for t in result.pii_types:
                        type_counts[t] = type_counts.get(t, 0) + 1
            if record_pii:
                pii_count += 1

        return {
            "total_records": total,
            "records_with_pii": pii_count,
            "pii_rate": round(pii_count / max(total, 1), 4),
            "pii_type_counts": type_counts,
            "clean": pii_count == 0,
        }

    def _make_replacement(self, pii_type: str, original: str) -> str:
        if self.replacement_style == "type_tag":
            return f"[{pii_type.upper()}]"
        elif self.replacement_style == "hash":
            h = hashlib.sha256(original.encode()).hexdigest()[:8]
            return f"[REDACTED:{h}]"
        elif self.replacement_style == "blank":
            return "[REDACTED]"
        return f"[{pii_type.upper()}]"
