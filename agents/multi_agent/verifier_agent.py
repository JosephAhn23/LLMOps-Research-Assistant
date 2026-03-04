"""
Verifier Agent: fact-checks individual claims against retrieved sources.

Each claim is classified as:
  VERIFIED: supported by retrieved context with high confidence
  PARTIALLY_SUPPORTED: some evidence but incomplete
  UNSUPPORTED: no supporting evidence found
  CONTRADICTED: evidence contradicts the claim

Returns a structured verification report with per-claim status,
supporting evidence, and an overall veracity score.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.multi_agent.base_agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    BaseAgent,
    ToolRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    claim: str
    status: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim[:120],
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "n_supporting": len(self.supporting_evidence),
            "n_contradicting": len(self.contradicting_evidence),
        }


class VerifierAgent(BaseAgent):
    """
    Verifies factual claims in research output against retrieved sources.

    Algorithm:
    1. Extract atomic claims from the answer (sentence-level)
    2. For each claim, compute keyword overlap with each context chunk
    3. Check for contradicting evidence (negation patterns)
    4. Classify claim status based on coverage thresholds
    5. Compute overall veracity score
    """

    VERIFIED_THRESHOLD = 0.40
    PARTIAL_THRESHOLD = 0.15

    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bfalse\b",
        r"\bincorrect\b", r"\bwrong\b", r"\bcontrary\b",
    ]

    def __init__(
        self,
        name: str = "verifier",
        tools: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        super().__init__(name, tools, timeout_seconds)

    def process(self, task: AgentTask) -> AgentResult:
        answer = task.context.get("research_result", "")
        context = task.context.get("retrieved_context", [])

        if not answer:
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error="No research result to verify.",
                confidence=0.0,
            )

        claims = self._extract_claims(answer)
        verifications = [self._verify_claim(c, context) for c in claims]

        n_verified = sum(1 for v in verifications if v.status == "VERIFIED")
        n_partial = sum(1 for v in verifications if v.status == "PARTIALLY_SUPPORTED")
        n_unsupported = sum(1 for v in verifications if v.status == "UNSUPPORTED")
        n_contradicted = sum(1 for v in verifications if v.status == "CONTRADICTED")

        veracity_score = (
            n_verified * 1.0 + n_partial * 0.5 - n_contradicted * 0.5
        ) / max(len(verifications), 1)
        veracity_score = max(0.0, min(1.0, veracity_score))

        summary = (
            f"{n_verified}/{len(verifications)} claims verified, "
            f"{n_partial} partially supported, "
            f"{n_unsupported} unsupported, "
            f"{n_contradicted} contradicted."
        )

        self.logger.info("Verification: task=%s %s", task.task_id, summary)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=AgentStatus.SUCCEEDED,
            output=summary,
            confidence=veracity_score,
            reasoning=f"Verified {len(claims)} claims against {len(context)} context chunks.",
            metadata={
                "verifications": [v.to_dict() for v in verifications],
                "veracity_score": round(veracity_score, 3),
                "n_verified": n_verified,
                "n_unsupported": n_unsupported,
                "n_contradicted": n_contradicted,
            },
        )

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip().split()) >= 5][:8]

    def _verify_claim(self, claim: str, context: List[Dict]) -> ClaimVerification:
        claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
        supporting = []
        contradicting = []

        for chunk in context:
            chunk_text = chunk.get("text", "")
            chunk_words = set(w.lower() for w in chunk_text.split() if len(w) > 3)
            if not chunk_words:
                continue

            overlap = len(claim_words & chunk_words) / max(len(claim_words), 1)

            if overlap >= self.VERIFIED_THRESHOLD:
                if self._has_negation_conflict(claim, chunk_text):
                    contradicting.append(chunk_text[:100])
                else:
                    supporting.append(chunk_text[:100])
            elif overlap >= self.PARTIAL_THRESHOLD:
                supporting.append(chunk_text[:100])

        if contradicting:
            status = "CONTRADICTED"
            confidence = 0.2
        elif len(supporting) >= 2:
            status = "VERIFIED"
            confidence = min(0.7 + 0.1 * len(supporting), 0.95)
        elif len(supporting) == 1:
            status = "PARTIALLY_SUPPORTED"
            confidence = 0.5
        else:
            status = "UNSUPPORTED"
            confidence = 0.15

        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
        )

    def _has_negation_conflict(self, claim: str, chunk_text: str) -> bool:
        claim_lower = claim.lower()
        chunk_lower = chunk_text.lower()
        claim_has_negation = any(re.search(p, claim_lower) for p in self.NEGATION_PATTERNS)
        chunk_has_negation = any(re.search(p, chunk_lower) for p in self.NEGATION_PATTERNS)
        return claim_has_negation != chunk_has_negation
