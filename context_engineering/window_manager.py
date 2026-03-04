"""
Context window budget manager.

Allocates the available token budget across competing context sources:
  - System prompt
  - Conversation history
  - Retrieved chunks
  - Few-shot examples
  - Scratch pad / chain-of-thought space

Uses priority-based eviction: lower-priority items are truncated first
when the total budget is exceeded.

Usage:
    manager = WindowManager(total_budget=8192, model="gpt-4o-mini")
    manager.add("system", system_prompt, priority=10)
    manager.add("history", conversation_history, priority=7)
    manager.add("chunks", retrieved_context, priority=9)
    manager.add("few_shot", few_shot_block, priority=5)

    result = manager.fit()
    print(result.summary())
    final_prompt = result.assembled_prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Model-specific token budgets (context window - generation reserve)
MODEL_BUDGETS: dict[str, int] = {
    "gpt-4o": 127_000,
    "gpt-4o-mini": 127_000,
    "gpt-4-turbo": 127_000,
    "gpt-3.5-turbo": 15_500,
    "claude-3-5-sonnet": 195_000,
    "llama-3.1-8b": 127_000,
    "mistral-7b": 31_500,
}

GENERATION_RESERVE = 1024  # tokens reserved for model output


def _count_tokens(text: str) -> int:
    """Rough token count (words * 4/3). Replace with tiktoken for precision."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text.split()) * 4 // 3)


@dataclass
class ContextSlot:
    name: str
    text: str
    priority: int
    token_count: int = 0
    truncated: bool = False
    original_token_count: int = 0

    def __post_init__(self) -> None:
        self.token_count = _count_tokens(self.text)
        self.original_token_count = self.token_count


@dataclass
class WindowResult:
    slots: list[ContextSlot]
    total_budget: int
    used_tokens: int
    truncated_slots: list[str]
    assembled_prompt: str

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.total_budget

    def summary(self) -> str:
        lines = [
            f"Window Budget: {self.used_tokens}/{self.total_budget} tokens "
            f"({self.utilization:.1%} utilized)",
        ]
        for slot in sorted(self.slots, key=lambda s: -s.priority):
            status = " [TRUNCATED]" if slot.truncated else ""
            lines.append(
                f"  {slot.name:15s} priority={slot.priority:2d}  "
                f"{slot.token_count:5d} tokens{status}"
            )
        if self.truncated_slots:
            lines.append(f"Truncated: {', '.join(self.truncated_slots)}")
        return "\n".join(lines)


class WindowManager:
    """
    Priority-based context window allocator.

    Parameters
    ----------
    total_budget : int | None
        Maximum tokens. If None, looks up model in MODEL_BUDGETS.
    model : str
        Model name for automatic budget lookup.
    generation_reserve : int
        Tokens reserved for model output (subtracted from total_budget).
    assembly_order : list[str] | None
        Slot names in the order they should appear in the assembled prompt.
        If None, slots are assembled in descending priority order.
    """

    def __init__(
        self,
        total_budget: int | None = None,
        model: str = "gpt-4o-mini",
        generation_reserve: int = GENERATION_RESERVE,
        assembly_order: list[str] | None = None,
    ) -> None:
        if total_budget is not None:
            self.total_budget = total_budget - generation_reserve
        else:
            raw = MODEL_BUDGETS.get(model, 127_000)
            self.total_budget = raw - generation_reserve

        self.assembly_order = assembly_order
        self._slots: list[ContextSlot] = []

    def add(self, name: str, text: str, priority: int = 5) -> "WindowManager":
        """
        Add a context slot.

        Parameters
        ----------
        name : str
            Identifier (e.g., "system", "history", "chunks").
        text : str
            The text content.
        priority : int
            Eviction priority: higher = kept longer. Range [1, 10].
        """
        self._slots.append(ContextSlot(name=name, text=text, priority=priority))
        return self

    def fit(self) -> WindowResult:
        """
        Fit all slots into the budget using priority-based eviction.

        Eviction strategy:
        1. Compute total tokens across all slots.
        2. While over budget, truncate the lowest-priority slot by 20%.
        3. Repeat until within budget or all slots at minimum size.
        """
        slots = [ContextSlot(s.name, s.text, s.priority) for s in self._slots]
        truncated_names: list[str] = []

        total = sum(s.token_count for s in slots)

        while total > self.total_budget:
            # Find lowest-priority slot with content to truncate
            evictable = [s for s in slots if s.token_count > 50]
            if not evictable:
                break
            target = min(evictable, key=lambda s: (s.priority, -s.token_count))

            # Truncate by 20% (word-level approximation)
            words = target.text.split()
            keep = max(int(len(words) * 0.8), 10)
            target.text = " ".join(words[:keep]) + " [...]"
            target.token_count = _count_tokens(target.text)
            target.truncated = True
            if target.name not in truncated_names:
                truncated_names.append(target.name)

            total = sum(s.token_count for s in slots)

        # Assemble prompt
        if self.assembly_order:
            slot_map = {s.name: s for s in slots}
            ordered = [slot_map[n] for n in self.assembly_order if n in slot_map]
            remainder = [s for s in slots if s.name not in self.assembly_order]
            ordered.extend(sorted(remainder, key=lambda s: -s.priority))
        else:
            ordered = sorted(slots, key=lambda s: -s.priority)

        assembled = "\n\n".join(s.text for s in ordered if s.text.strip())

        logger.info(
            "WindowManager: %d/%d tokens used (%.1f%%), %d slots truncated",
            total, self.total_budget, 100 * total / self.total_budget,
            len(truncated_names),
        )

        return WindowResult(
            slots=slots,
            total_budget=self.total_budget,
            used_tokens=total,
            truncated_slots=truncated_names,
            assembled_prompt=assembled,
        )

    def clear(self) -> "WindowManager":
        self._slots = []
        return self

    @classmethod
    def for_model(
        cls,
        model: str,
        assembly_order: list[str] | None = None,
    ) -> "WindowManager":
        """Factory: create a WindowManager sized for a specific model."""
        return cls(model=model, assembly_order=assembly_order)
