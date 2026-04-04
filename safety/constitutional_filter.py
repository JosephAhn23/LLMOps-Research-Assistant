"""
API gateway constitutional gate — behavioral + Mitchell-style pre-inference checks.

Combines :mod:`safety.behavioral_classifier` with lightweight query policies aligned
to model-card practice (Mitchell et al., 2019): block clearly out-of-scope or
manipulative requests before expensive LLM work.

See also :mod:`governance.model_card_generator` and :mod:`governance.constitution`
for documentation-grade artifacts; this module is optimized for **latency** on
every HTTP request.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from safety.behavioral_classifier import STANDARD_REFUSAL, classify_prompt_heuristic

logger = logging.getLogger(__name__)

# Model-card inspired: refuse encoded / exfiltration-style payloads at the edge
_OVERSIZED_BYTES = int(os.getenv("GATEWAY_MAX_QUERY_BYTES", "12000"))
_SUSPICIOUS_ENCODED = re.compile(r"(?:\\x[0-9a-fA-F]{2}){8,}|%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}")


@dataclass
class ConstitutionalFilterResult:
    allowed: bool
    reason: str
    refusal_message: str = ""

    def to_http_detail(self) -> str:
        return self.refusal_message or self.reason


def constitutional_filter_query(text: str) -> ConstitutionalFilterResult:
    """
    Fast synchronous gate: injection/toxicity heuristics + size / encoding checks.
    """
    raw = text or ""
    if len(raw.encode("utf-8")) > _OVERSIZED_BYTES:
        return ConstitutionalFilterResult(
            False,
            "query_too_large",
            "Request text exceeds gateway size limits.",
        )
    if _SUSPICIOUS_ENCODED.search(raw):
        return ConstitutionalFilterResult(
            False,
            "suspicious_encoding",
            "Request contains suspicious encoded content and was blocked.",
        )

    api_thr = float(os.getenv("API_BEHAVIORAL_BLOCK_THRESHOLD", "0.72"))
    beh = classify_prompt_heuristic(raw, block_threshold=api_thr)
    if beh.blocked:
        return ConstitutionalFilterResult(
            False,
            "behavioral_block",
            beh.refusal_message or STANDARD_REFUSAL,
        )

    return ConstitutionalFilterResult(True, "ok", "")


def audit_gateway_decision(
    *,
    endpoint: str,
    allowed: bool,
    reason: str,
    actor: str = "api_gateway",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append to SHA-256 audit chain when ``AUDIT_LOG_PATH`` is set."""
    path = os.getenv("AUDIT_LOG_PATH", "").strip()
    if not path:
        return
    try:
        from governance.audit_log import CryptoAuditLog

        CryptoAuditLog(persist_path=path).log(
            event_type="gateway_constitutional_gate",
            actor=actor,
            payload={
                "endpoint": endpoint,
                "allowed": allowed,
                "reason": reason,
                **(extra or {}),
            },
        )
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)


class ConstitutionalFilter:
    """Callable facade for FastAPI dependencies."""

    def check(self, text: str) -> ConstitutionalFilterResult:
        return constitutional_filter_query(text)
