"""
Cross-provider "truth committee" consensus.

Runs the same prompt on two independent LLM backends (e.g. OpenAI + Anthropic),
compares answers, and flags human-in-the-loop when they materially disagree.

Designed for high-stakes numeric or factual checks; agreement uses normalized
text similarity by default, with an optional third-party LLM judge.
"""
from __future__ import annotations

import concurrent.futures
import difflib
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from agents.multi_agent.base_agent import AgentResult, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class ProviderAnswer:
    """Raw completion from one backend."""

    provider_id: str
    model: str
    text: str
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass
class CrossProviderConsensusResult:
    """Outcome of a two-provider consensus run."""

    task_id: str
    prompt_excerpt: str
    answers: Tuple[ProviderAnswer, ProviderAnswer]
    agreement_score: float
    models_agree: bool
    final_text: str
    strategy: str
    hitl_required: bool
    hitl_reason: str
    judge_rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_agent_results(self) -> List[AgentResult]:
        """Map provider answers to AgentResult for the existing consensus stack."""
        out: List[AgentResult] = []
        for pa in self.answers:
            status = AgentStatus.SUCCEEDED if pa.ok else AgentStatus.FAILED
            conf = 1.0 if pa.ok and self.models_agree else (0.85 if pa.ok else 0.0)
            out.append(
                AgentResult(
                    task_id=self.task_id,
                    agent_name=pa.provider_id,
                    status=status,
                    output=pa.text,
                    confidence=conf,
                    latency_ms=pa.latency_ms,
                    error=pa.error,
                    metadata={"model": pa.model, "truth_committee": True},
                )
            )
        return out


class TruthCommitteeOutcome(BaseModel):
    """
    Pydantic summary for the LangGraph / API quality gate.

    Aligns with the conceptual CONSENSUS: TRUE/FALSE flow: ``is_consensus_reached``
    is True only when two providers agree (or a judge reconciles them) without HITL.
    """

    final_answer: str = Field(description="Answer shown to the user after the gate.")
    is_consensus_reached: bool
    discrepancy_notes: str | None = Field(
        default=None,
        description="Why consensus failed or judge commentary.",
    )
    agreement_score: float = 0.0
    strategy: str = ""

    @classmethod
    def from_gate(
        cls,
        raw: CrossProviderConsensusResult,
        presented_answer: str,
    ) -> TruthCommitteeOutcome:
        reached = raw.models_agree and not raw.hitl_required
        notes: str | None = None
        if raw.hitl_required:
            notes = (raw.hitl_reason or "").strip() or None
            if raw.judge_rationale:
                notes = (notes + "\n" + raw.judge_rationale).strip() if notes else raw.judge_rationale
        elif raw.judge_rationale:
            notes = raw.judge_rationale
        return cls(
            final_answer=presented_answer,
            is_consensus_reached=reached,
            discrepancy_notes=notes,
            agreement_score=raw.agreement_score,
            strategy=raw.strategy,
        )


class LLMProvider(ABC):
    """Minimal interface for any chat-style model."""

    provider_id: str
    model: str

    @abstractmethod
    def complete(self, system: str, user: str) -> ProviderAnswer:
        """Return assistant text and record latency."""


def _normalize_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s.,+\-*/=^()%\[\]]+", "", t)
    return t


def text_agreement_score(a: str, b: str) -> float:
    """Token- and character-level blend; 1.0 = identical after normalization."""
    na, nb = _normalize_answer(a), _normalize_answer(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    char = difflib.SequenceMatcher(a=na, b=nb).ratio()
    wa, wb = set(na.split()), set(nb.split())
    if not wa and not wb:
        return 1.0
    jacc = len(wa & wb) / max(len(wa | wb), 1)
    return float(0.5 * char + 0.5 * jacc)


class OpenAIChatProvider(LLMProvider):
    """OpenAI-compatible Chat Completions (OpenAI, Azure OpenAI, local gateways)."""

    def __init__(
        self,
        provider_id: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
    ):
        from openai import OpenAI

        self.provider_id = provider_id
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s or float(os.getenv("OPENAI_TIMEOUT_S", "60"))
        kwargs: Dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> ProviderAnswer:
        t0 = time.perf_counter()
        try:
            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
            text = (resp.choices[0].message.content or "").strip()
            return ProviderAnswer(
                provider_id=self.provider_id,
                model=self.model,
                text=text,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            logger.warning("%s completion failed: %s", self.provider_id, exc)
            return ProviderAnswer(
                provider_id=self.provider_id,
                model=self.model,
                text="",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )


class AnthropicMessagesProvider(LLMProvider):
    """Anthropic Messages API (requires `pip install anthropic`)."""

    def __init__(
        self,
        provider_id: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the anthropic package: pip install anthropic"
            ) from exc

        self.provider_id = provider_id
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s or float(os.getenv("ANTHROPIC_TIMEOUT_S", "120"))
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s)

    def complete(self, system: str, user: str) -> ProviderAnswer:
        t0 = time.perf_counter()
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": user}],
                "temperature": self.temperature,
            }
            if system.strip():
                kwargs["system"] = system
            msg = self._client.messages.create(**kwargs)
            parts: List[str] = []
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            text = "".join(parts).strip()
            return ProviderAnswer(
                provider_id=self.provider_id,
                model=self.model,
                text=text,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            logger.warning("%s completion failed: %s", self.provider_id, exc)
            return ProviderAnswer(
                provider_id=self.provider_id,
                model=self.model,
                text="",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )


class CrossProviderConsensusNode:
    """
    Run two providers on the same prompt; require agreement or escalate to HITL.

    agreement_threshold: minimum blended similarity (see text_agreement_score) to
      treat answers as agreeing without a judge.
    judge: optional callable (answer_a, answer_b, prompt) -> (agree: bool, rationale).
      If provided and similarity is below threshold, the judge decides; if the judge
      says disagree, HITL is required.
    """

    def __init__(
        self,
        provider_a: LLMProvider,
        provider_b: LLMProvider,
        *,
        agreement_threshold: float = 0.82,
        judge: Optional[Callable[[str, str, str], Tuple[bool, str]]] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ):
        self.provider_a = provider_a
        self.provider_b = provider_b
        self.agreement_threshold = agreement_threshold
        self.judge = judge
        self._executor = executor

    def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        *,
        task_id: Optional[str] = None,
    ) -> CrossProviderConsensusResult:
        tid = task_id or str(uuid.uuid4())[:12]
        excerpt = user_prompt[:200] + ("…" if len(user_prompt) > 200 else "")

        def _call(p: LLMProvider) -> ProviderAnswer:
            return p.complete(system_prompt, user_prompt)

        if self._executor is not None:
            fut_a = self._executor.submit(_call, self.provider_a)
            fut_b = self._executor.submit(_call, self.provider_b)
            ans_a = fut_a.result()
            ans_b = fut_b.result()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(_call, self.provider_a)
                fut_b = ex.submit(_call, self.provider_b)
                ans_a = fut_a.result()
                ans_b = fut_b.result()

        if not ans_a.ok and not ans_b.ok:
            return CrossProviderConsensusResult(
                task_id=tid,
                prompt_excerpt=excerpt,
                answers=(ans_a, ans_b),
                agreement_score=0.0,
                models_agree=False,
                final_text="",
                strategy="both_failed",
                hitl_required=True,
                hitl_reason="Both providers failed; human review required.",
                metadata={"errors": (ans_a.error, ans_b.error)},
            )

        if not ans_a.ok:
            return CrossProviderConsensusResult(
                task_id=tid,
                prompt_excerpt=excerpt,
                answers=(ans_a, ans_b),
                agreement_score=0.0,
                models_agree=False,
                final_text=ans_b.text,
                strategy="single_provider_fallback",
                hitl_required=True,
                hitl_reason=f"Provider {ans_a.provider_id} failed; using {ans_b.provider_id} only.",
                metadata={"fallback": ans_b.provider_id, "error": ans_a.error},
            )

        if not ans_b.ok:
            return CrossProviderConsensusResult(
                task_id=tid,
                prompt_excerpt=excerpt,
                answers=(ans_a, ans_b),
                agreement_score=0.0,
                models_agree=False,
                final_text=ans_a.text,
                strategy="single_provider_fallback",
                hitl_required=True,
                hitl_reason=f"Provider {ans_b.provider_id} failed; using {ans_a.provider_id} only.",
                metadata={"fallback": ans_a.provider_id, "error": ans_b.error},
            )

        score = text_agreement_score(ans_a.text, ans_b.text)
        agree = score >= self.agreement_threshold
        rationale = ""
        strategy = "similarity"

        if not agree and self.judge is not None:
            judge_agree, rationale = self.judge(ans_a.text, ans_b.text, user_prompt)
            agree = judge_agree
            strategy = "llm_judge"

        final = ans_a.text if len(ans_a.text) >= len(ans_b.text) else ans_b.text
        if agree:
            final = ans_a.text.strip()

        hitl = not agree
        reason = ""
        if hitl:
            reason = (
                f"Models disagree (similarity={score:.3f}, threshold={self.agreement_threshold}). "
                "Human review required."
            )

        return CrossProviderConsensusResult(
            task_id=tid,
            prompt_excerpt=excerpt,
            answers=(ans_a, ans_b),
            agreement_score=score,
            models_agree=agree,
            final_text=final if agree else "",
            strategy=strategy,
            hitl_required=hitl,
            hitl_reason=reason,
            judge_rationale=rationale,
            metadata={"threshold": self.agreement_threshold},
        )


def openai_judge_factory(model: str = "gpt-4o-mini") -> Callable[[str, str, str], Tuple[bool, str]]:
    """
    LLM-as-judge using the default OpenAI client.
    Returns (agree, short_rationale).
    """
    provider = OpenAIChatProvider("consensus_judge", model, temperature=0.0, max_tokens=256)

    def judge(a: str, b: str, prompt: str) -> Tuple[bool, str]:
        sys = (
            "You compare two model answers to the same question. "
            "Decide if they are materially the same conclusion (including numerics). "
            "Reply with exactly one line: AGREE or DISAGREE, then a short reason."
        )
        user = (
            f"Question:\n{prompt}\n\n--- Answer A ---\n{a}\n\n--- Answer B ---\n{b}\n"
        )
        pa = provider.complete(sys, user)
        if not pa.ok:
            return False, f"judge_error:{pa.error}"
        line = pa.text.strip().split("\n")[0].upper()
        agree = line.startswith("AGREE")
        return agree, pa.text.strip()[:500]

    return judge


def default_truth_committee_from_env() -> Optional[CrossProviderConsensusNode]:
    """
    If both OPENAI_API_KEY and ANTHROPIC_API_KEY are set, build OpenAI + Anthropic node.

    Model names from CONSENSUS_OPENAI_MODEL and CONSENSUS_ANTHROPIC_MODEL.
    Returns None if Anthropic is unavailable or keys missing.
    """
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        oa = OpenAIChatProvider(
            "openai",
            os.getenv("CONSENSUS_OPENAI_MODEL", "gpt-4o-mini"),
        )
        cl = AnthropicMessagesProvider(
            "anthropic",
            os.getenv("CONSENSUS_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        )
    except Exception as exc:
        logger.info("Truth committee not configured: %s", exc)
        return None
    return CrossProviderConsensusNode(oa, cl, judge=openai_judge_factory())
