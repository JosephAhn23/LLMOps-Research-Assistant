"""
Economic guardrails: token/time estimates vs budget → HITL when ROI is unclear.

Pauses high-burn work when estimated iterations or API spend exceed configurable
thresholds (default: 5 iterations, $2.00).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class EconomicVerdict:
    """Outcome of an economic pre-flight check."""

    requires_hitl: bool
    estimated_usd: float
    estimated_iterations: int
    rationale: str
    roi_prompt: str
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requires_hitl": self.requires_hitl,
            "estimated_usd": round(self.estimated_usd, 4),
            "estimated_iterations": self.estimated_iterations,
            "rationale": self.rationale,
            "roi_prompt": self.roi_prompt,
            "flags": list(self.flags),
        }


class EconomicJudge:
    """
    Compare estimated cost to caps; emit HITL when spend or iteration risk is high.

    Pricing is approximate: set ``ECONOMIC_PRICE_PER_1K_INPUT_TOKENS`` and
    ``ECONOMIC_PRICE_PER_1K_OUTPUT_TOKENS`` to match your model (e.g. Opus vs Haiku).
    """

    def __init__(
        self,
        *,
        max_iterations: Optional[int] = None,
        max_usd_per_task: Optional[float] = None,
        price_per_1k_input: Optional[float] = None,
        price_per_1k_output: Optional[float] = None,
    ) -> None:
        self.max_iterations = (
            max_iterations if max_iterations is not None else _i("ECONOMIC_JUDGE_MAX_ITERATIONS", 5)
        )
        self.max_usd_per_task = (
            max_usd_per_task if max_usd_per_task is not None else _f("ECONOMIC_JUDGE_MAX_USD", 2.0)
        )
        self.price_per_1k_input = (
            price_per_1k_input
            if price_per_1k_input is not None
            else _f("ECONOMIC_PRICE_PER_1K_INPUT_TOKENS", 0.015)
        )
        self.price_per_1k_output = (
            price_per_1k_output
            if price_per_1k_output is not None
            else _f("ECONOMIC_PRICE_PER_1K_OUTPUT_TOKENS", 0.075)
        )

    def estimate_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (max(0, input_tokens) / 1000.0) * self.price_per_1k_input
            + (max(0, output_tokens) / 1000.0) * self.price_per_1k_output
        )

    def evaluate(
        self,
        *,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        estimated_iterations: int = 1,
        task_label: str = "",
        value_note: str = "",
    ) -> EconomicVerdict:
        flags: List[str] = []
        usd = self.estimate_usd(estimated_input_tokens, estimated_output_tokens) * max(
            1, estimated_iterations
        )

        hitl_iter = estimated_iterations > self.max_iterations
        hitl_usd = usd > self.max_usd_per_task
        if hitl_iter:
            flags.append("iteration_cap")
        if hitl_usd:
            flags.append("usd_cap")

        requires_hitl = hitl_iter or hitl_usd
        rationale_parts = []
        if hitl_iter:
            rationale_parts.append(
                f"Estimated {estimated_iterations} iterations exceed cap {self.max_iterations}."
            )
        if hitl_usd:
            rationale_parts.append(
                f"Estimated ${usd:.2f} exceeds cap ${self.max_usd_per_task:.2f}."
            )
        if not requires_hitl:
            rationale_parts.append("Within iteration and spend caps.")

        roi = (
            f"Is '{task_label or 'this task'}' worth ~${usd:.2f} and ~{estimated_iterations} agent iterations? "
            f"{value_note}".strip()
        )
        if requires_hitl:
            roi += (
                " Approve to proceed, or narrow scope (smaller prompt, cheaper model, or split work)."
            )

        return EconomicVerdict(
            requires_hitl=requires_hitl,
            estimated_usd=usd,
            estimated_iterations=estimated_iterations,
            rationale=" ".join(rationale_parts),
            roi_prompt=roi,
            flags=flags,
        )
