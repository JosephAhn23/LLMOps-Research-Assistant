"""
ConsensusNode — Schwartz-style adversarial consensus with self-correction.

Runs two providers in parallel (e.g. Claude + GPT-4o), optionally with a judge,
and on disagreement appends a **self-correction** block before retrying (instead
of immediately halting), up to ``max_self_corrections``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from agents.multi_agent.cross_provider_consensus import (
    AnthropicMessagesProvider,
    CrossProviderConsensusNode,
    CrossProviderConsensusResult,
    LLMProvider,
    OpenAIChatProvider,
    openai_judge_factory,
)

logger = logging.getLogger(__name__)


class ConsensusState(str, Enum):
    """High-level outcome for orchestrators and LangGraph routing."""

    CONSENSUS_OK = "CONSENSUS_OK"
    DISCREPANCY_DETECTED = "DISCREPANCY_DETECTED"
    BOTH_FAILED = "BOTH_FAILED"


@dataclass
class ConsensusNodeResult:
    state: ConsensusState
    final_answer: str
    attempts: int
    discrepancy_notes: Optional[str] = None
    last_raw: Optional[CrossProviderConsensusResult] = None
    self_correction_trace: List[str] = field(default_factory=list)


def _self_correction_block(raw: CrossProviderConsensusResult) -> str:
    a, b = raw.answers
    return (
        "\n\n--- SELF-CORRECTION ROUND ---\n"
        "Two independent models disagreed on facts or logic. Produce ONE reconciled answer.\n"
        "- Show reasoning compatible with both outputs where possible.\n"
        "- If the conflict cannot be resolved from the prompt alone, state what is uncertain.\n"
        f"- Model A ({a.provider_id}): {a.text[:6000]}\n"
        f"- Model B ({b.provider_id}): {b.text[:6000]}\n"
        f"- Automated similarity score: {raw.agreement_score:.3f}\n"
        f"- Prior note: {raw.hitl_reason}\n"
    )


class ConsensusNode:
    """
    Truth committee with optional **retry** after ``DISCREPANCY_DETECTED``.

    Wraps :class:`CrossProviderConsensusNode`; same parallel execution and judge
    behavior, plus up to ``max_self_corrections`` additional rounds that append
    a structured reconciliation prompt.
    """

    def __init__(
        self,
        primary: LLMProvider,
        secondary: LLMProvider,
        *,
        agreement_threshold: float = 0.82,
        use_openai_judge: bool = True,
        judge_model: str = "gpt-4o-mini",
        max_self_corrections: int = 1,
    ) -> None:
        judge = openai_judge_factory(judge_model) if use_openai_judge else None
        self._committee = CrossProviderConsensusNode(
            primary,
            secondary,
            agreement_threshold=agreement_threshold,
            judge=judge,
        )
        self.max_self_corrections = max_self_corrections

    @classmethod
    def from_default_providers(
        cls,
        *,
        max_self_corrections: int = 1,
        agreement_threshold: float = 0.82,
    ) -> ConsensusNode:
        """
        Claude (Anthropic) as primary, GPT-4o-mini as secondary — matches common setups.

        Requires ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` and ``pip install anthropic``.
        """
        openai_model = os.getenv("CONSENSUS_NODE_OPENAI_MODEL", "gpt-4o-mini")
        anthropic_model = os.getenv("CONSENSUS_NODE_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        primary = AnthropicMessagesProvider("claude", anthropic_model)
        secondary = OpenAIChatProvider("openai", openai_model)
        return cls(
            primary,
            secondary,
            agreement_threshold=agreement_threshold,
            use_openai_judge=True,
            judge_model=openai_model,
            max_self_corrections=max_self_corrections,
        )

    def run(self, prompt: str, system: str = "") -> ConsensusNodeResult:
        trace: List[str] = []
        user = prompt
        last: Optional[CrossProviderConsensusResult] = None

        max_attempts = 1 + max(0, self.max_self_corrections)
        for attempt in range(1, max_attempts + 1):
            last = self._committee.run(user, system)
            if last.strategy == "both_failed":
                return ConsensusNodeResult(
                    state=ConsensusState.BOTH_FAILED,
                    final_answer="",
                    attempts=attempt,
                    discrepancy_notes=last.hitl_reason,
                    last_raw=last,
                    self_correction_trace=trace,
                )
            if not last.hitl_required:
                text = last.final_text.strip() or last.answers[0].text.strip()
                return ConsensusNodeResult(
                    state=ConsensusState.CONSENSUS_OK,
                    final_answer=text,
                    attempts=attempt,
                    last_raw=last,
                    self_correction_trace=trace,
                )
            if attempt < max_attempts:
                block = _self_correction_block(last)
                trace.append(block[:2000])
                user = prompt + block
                logger.info(
                    "ConsensusNode: discrepancy on attempt %d/%d — retrying with self-correction",
                    attempt,
                    max_attempts,
                )
            else:
                notes = last.hitl_reason or "DISCREPANCY_DETECTED"
                return ConsensusNodeResult(
                    state=ConsensusState.DISCREPANCY_DETECTED,
                    final_answer="",
                    attempts=attempt,
                    discrepancy_notes=notes,
                    last_raw=last,
                    self_correction_trace=trace,
                )

        raise RuntimeError("ConsensusNode.run: unreachable")
