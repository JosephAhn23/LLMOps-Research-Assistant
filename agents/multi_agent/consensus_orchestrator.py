"""
Adversarial consensus — dual independent researchers, numeric verifier, skeptic, hard reset.

Implements a three-role loop aligned with “researcher / skeptic / verifier” patterns:
two models answer the same RAG-backed prompt in **parallel**; the **verifier** compares
extracted numeric claims (relative error > 0.1% → conflict). On conflict we log a
**Conflict** event to MLflow and apply a **hard reset** (re-prompt primary model to
re-derive from first principles). A **skeptic** model then reviews the surviving answer
for reward-hacking / logical leaps.

LangGraph-shaped state (for copying into a custom graph)::

    class AdversarialConsensusState(TypedDict, total=False):
        query: str
        rag_context: str
        researcher_primary_text: str
        researcher_secondary_text: str
        numeric_conflict: bool
        numeric_conflict_detail: str
        skeptic_review: str
        reset_round: int
        final_answer: str
        conflict_events: List[str]
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, NotRequired, Optional, Tuple, TypedDict

from agents.multi_agent.cross_provider_consensus import (
    AnthropicMessagesProvider,
    LLMProvider,
    OpenAIChatProvider,
    ProviderAnswer,
)
from mlops.compat import mlflow

logger = logging.getLogger(__name__)

# Relative tolerance: 0.001 == 0.1% of max(|a|, |b|, ε)
DEFAULT_NUMERIC_TOLERANCE = float(os.getenv("ADVERSARIAL_NUMERIC_TOLERANCE", "0.001"))
DEFAULT_MAX_RESETS = int(os.getenv("ADVERSARIAL_MAX_HARD_RESETS", "2"))

_RE_FLOAT = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})*(?:\.\d*)?|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


class AdversarialConsensusState(TypedDict, total=False):
    """State bag for a LangGraph implementation of this loop (reference schema)."""

    query: str
    rag_context: str
    researcher_primary_text: str
    researcher_secondary_text: str
    numeric_conflict: bool
    numeric_conflict_detail: str
    skeptic_review: str
    reset_round: int
    final_answer: str
    conflict_events: List[str]
    done: NotRequired[bool]


RESEARCHER_SYSTEM = (
    "You are a precise research assistant. Answer using ONLY the provided context. "
    "Cite sources as [source_N]. Give explicit numeric results when the question requires them."
)

SKEPTIC_SYSTEM = (
    "You are an adversarial reviewer. Find reward hacking, unjustified logical leaps, "
    "and claims not supported by the context. Reply with one line: "
    "VERDICT: SOUND or VERDICT: UNSOUND — then brief reasons. "
    "If UNSOUND, start the reasons line with FLAG: REWARD_HACK or FLAG: UNSUPPORTED."
)

HARD_RESET_USER_SUFFIX = (
    "\n\n[SYSTEM NOTE] An independent model produced numerically inconsistent results. "
    "Re-derive from first principles using only the context; restate assumptions explicitly."
)


def extract_floats(text: str) -> List[float]:
    """Parse numeric literals (scientific notation supported)."""
    scrubbed = re.sub(r"\[source_\d+\]", " ", text or "", flags=re.I)
    out: List[float] = []
    for m in _RE_FLOAT.finditer(scrubbed):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def numeric_relative_conflict(
    text_a: str,
    text_b: str,
    *,
    relative_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> Tuple[bool, str]:
    """
    True if both texts contain numbers and any paired (sorted) value differs beyond
    ``relative_tolerance * max(|x|,|y|,1e-12)``.
    If only one side has numbers, treat as conflict (cannot verify agreement).
    """
    fa = sorted(extract_floats(text_a))
    fb = sorted(extract_floats(text_b))
    if not fa and not fb:
        return False, ""
    if (not fa) ^ (not fb):
        return True, "numeric literals present in only one answer"
    if len(fa) != len(fb):
        return True, f"numeric count mismatch: {len(fa)} vs {len(fb)}"
    eps_floor = 1e-12
    for x, y in zip(fa, fb):
        scale = max(abs(x), abs(y), eps_floor)
        if abs(x - y) > relative_tolerance * scale:
            return True, f"values {x!r} vs {y!r} exceed relative tolerance {relative_tolerance}"
    return False, ""


def _skeptic_unsound(review: str) -> bool:
    t = (review or "").upper()
    return "VERDICT: UNSOUND" in t or "FLAG: REWARD_HACK" in t or "FLAG: UNSUPPORTED" in t


def _log_conflict_mlflow(detail: str, *, reset_round: int) -> None:
    try:
        mlflow.log_param("adversarial_consensus_event", "conflict")
        mlflow.log_metric("adversarial_conflict", 1.0)
        mlflow.log_param("adversarial_conflict_detail", detail[:500])
        mlflow.log_metric("adversarial_reset_round", float(reset_round))
    except Exception as exc:
        logger.debug("MLflow adversarial conflict log skipped: %s", exc)


def _parallel_complete(
    a: LLMProvider,
    b: LLMProvider,
    system: str,
    user: str,
) -> Tuple[ProviderAnswer, ProviderAnswer]:
    def _one(p: LLMProvider) -> ProviderAnswer:
        return p.complete(system, user)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_one, a)
        fb = ex.submit(_one, b)
        return fa.result(), fb.result()


@dataclass
class AdversarialConsensusOutcome:
    final_answer: str
    researcher_a: str
    researcher_b: str
    skeptic_review: str
    conflict_events: List[str] = field(default_factory=list)
    reset_count: int = 0
    hitl_recommended: bool = False

    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "researcher_a": self.researcher_a[:4000],
            "researcher_b": self.researcher_b[:4000],
            "skeptic_review": self.skeptic_review[:4000],
            "conflict_events": list(self.conflict_events),
            "reset_count": self.reset_count,
            "hitl_recommended": self.hitl_recommended,
        }


class ConsensusOrchestrator:
    """
    Adversarial multi-model loop with numeric verification and MLflow conflict logging.
    """

    def __init__(
        self,
        researcher_primary: LLMProvider,
        researcher_secondary: LLMProvider,
        skeptic: LLMProvider,
        *,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
        max_hard_resets: int = DEFAULT_MAX_RESETS,
    ) -> None:
        self.researcher_primary = researcher_primary
        self.researcher_secondary = researcher_secondary
        self.skeptic = skeptic
        self.numeric_tolerance = numeric_tolerance
        self.max_hard_resets = max_hard_resets

    @classmethod
    def from_env(cls) -> Optional["ConsensusOrchestrator"]:
        """Build OpenAI + Anthropic researchers and OpenAI skeptic when keys allow."""
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
            return None
        try:
            oa = OpenAIChatProvider(
                "openai_researcher",
                os.getenv("ADVERSARIAL_OPENAI_MODEL", "gpt-4o-mini"),
            )
            cl = AnthropicMessagesProvider(
                "anthropic_researcher",
                os.getenv("ADVERSARIAL_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )
            sk = OpenAIChatProvider(
                "openai_skeptic",
                os.getenv("ADVERSARIAL_SKEPTIC_MODEL", "gpt-4o-mini"),
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("ConsensusOrchestrator.from_env failed: %s", exc)
            return None
        return cls(oa, cl, sk)

    def run(self, query: str, rag_context: str) -> AdversarialConsensusOutcome:
        conflict_events: List[str] = []
        reset_count = 0
        user_base = f"Context:\n{rag_context}\n\nQuestion: {query}"
        extra = ""

        ans_a = ans_b = ""
        while True:
            user = user_base + extra
            pa, pb = _parallel_complete(
                self.researcher_primary,
                self.researcher_secondary,
                RESEARCHER_SYSTEM,
                user,
            )
            ans_a = (pa.text or "").strip()
            ans_b = (pb.text or "").strip()
            if not ans_a and not ans_b:
                return AdversarialConsensusOutcome(
                    final_answer="[Adversarial consensus] Both researchers failed.",
                    researcher_a="",
                    researcher_b="",
                    skeptic_review="",
                    conflict_events=["both_failed"],
                    reset_count=reset_count,
                    hitl_recommended=True,
                )

            bad, detail = numeric_relative_conflict(
                ans_a,
                ans_b,
                relative_tolerance=self.numeric_tolerance,
            )
            if bad:
                conflict_events.append(detail)
                _log_conflict_mlflow(detail, reset_round=reset_count)
                reset_count += 1
                if reset_count > self.max_hard_resets:
                    return AdversarialConsensusOutcome(
                        final_answer=(
                            "[Adversarial consensus] Numeric conflict persists after hard resets. "
                            "Human review required."
                        ),
                        researcher_a=ans_a,
                        researcher_b=ans_b,
                        skeptic_review="",
                        conflict_events=conflict_events,
                        reset_count=reset_count,
                        hitl_recommended=True,
                    )
                extra = HARD_RESET_USER_SUFFIX
                continue

            candidate = ans_a if len(ans_a) >= len(ans_b) else ans_b
            if not candidate:
                candidate = ans_a or ans_b

            sk_user = f"Context:\n{rag_context}\n\nProposed answer:\n{candidate}"
            sk = self.skeptic.complete(SKEPTIC_SYSTEM, sk_user)
            review = (sk.text or "").strip()

            if _skeptic_unsound(review):
                conflict_events.append(f"skeptic_unsound: {review[:200]}")
                _log_conflict_mlflow("skeptic_unsound", reset_round=reset_count)
                reset_count += 1
                if reset_count > self.max_hard_resets:
                    return AdversarialConsensusOutcome(
                        final_answer=(
                            "[Adversarial consensus] Skeptic flagged issues after max resets. "
                            "Human review required."
                        ),
                        researcher_a=ans_a,
                        researcher_b=ans_b,
                        skeptic_review=review,
                        conflict_events=conflict_events,
                        reset_count=reset_count,
                        hitl_recommended=True,
                    )
                extra = HARD_RESET_USER_SUFFIX
                continue

            return AdversarialConsensusOutcome(
                final_answer=candidate,
                researcher_a=ans_a,
                researcher_b=ans_b,
                skeptic_review=review,
                conflict_events=conflict_events,
                reset_count=reset_count,
                hitl_recommended=False,
            )
