"""
context_engineering/window_manager.py
---------------------------------------
Manages the LLM context window under a token budget.

Priority eviction: when context exceeds budget, lower-priority slots are
dropped first. Supports:
  - system prompt (highest priority, never evicted)
  - retrieved chunks (medium priority, evictable)
  - conversation history (low priority, oldest evicted first)
  - few-shot examples (configurable priority)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    SYSTEM = 100    # never evicted
    FEW_SHOT = 80   # evict last among optional
    RETRIEVED = 60  # evict if over budget
    HISTORY = 40    # evict oldest first
    SCRATCH = 20    # ephemeral, first to go


@dataclass
class ContextSlot:
    key: str
    text: str
    priority: Priority
    tokens: int = 0
    position: int = 0  # insertion order for FIFO eviction

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = max(1, len(self.text) // 4)


@dataclass
class WindowResult:
    slots: list[ContextSlot]
    total_tokens: int
    evicted: list[ContextSlot]
    budget: int

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.budget if self.budget else 0.0

    def to_messages(self) -> list[dict]:
        """Format slots as OpenAI-style messages list."""
        messages = []
        for slot in sorted(self.slots, key=lambda s: (-s.priority, s.position)):
            role = "system" if slot.priority == Priority.SYSTEM else "user"
            messages.append({"role": role, "content": slot.text})
        return messages

    def summary(self) -> str:
        lines = [
            f"Budget: {self.total_tokens}/{self.budget} ({self.utilization:.1%})",
            f"Slots kept: {len(self.slots)}, evicted: {len(self.evicted)}",
        ]
        for slot in self.slots:
            lines.append(f"  [{slot.priority.name:10s}] {slot.key}: {slot.tokens} tokens")
        if self.evicted:
            lines.append("Evicted:")
            for slot in self.evicted:
                lines.append(f"  [{slot.priority.name:10s}] {slot.key}: {slot.tokens} tokens")
        return "\n".join(lines)


class ContextWindowManager:
    """
    Assembles LLM context within a strict token budget.

    Usage
    -----
    >>> mgr = ContextWindowManager(token_budget=4096)
    >>> mgr.add("system", system_prompt, Priority.SYSTEM)
    >>> mgr.add("chunk_0", retrieved_text, Priority.RETRIEVED)
    >>> mgr.add("history", conversation, Priority.HISTORY)
    >>> result = mgr.build()
    >>> messages = result.to_messages()
    """

    def __init__(
        self,
        token_budget: int = 4096,
        reserve_for_output: int = 512,
        tokenizer: str | None = None,
    ):
        self.token_budget = token_budget
        self.reserve_for_output = reserve_for_output
        self._effective_budget = token_budget - reserve_for_output
        self._slots: list[ContextSlot] = []
        self._counter = 0
        self._tokenizer = None
        self._tokenizer_name = tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, key: str, text: str, priority: Priority = Priority.RETRIEVED) -> None:
        """Add a context slot. Duplicate keys overwrite existing."""
        tokens = self._count_tokens(text)
        slot = ContextSlot(
            key=key,
            text=text,
            priority=priority,
            tokens=tokens,
            position=self._counter,
        )
        # overwrite if key exists
        self._slots = [s for s in self._slots if s.key != key]
        self._slots.append(slot)
        self._counter += 1

    def add_chunks(
        self,
        chunks: list[str],
        prefix: str = "chunk",
        priority: Priority = Priority.RETRIEVED,
    ) -> None:
        """Add multiple retrieved chunks with auto-generated keys."""
        for i, chunk in enumerate(chunks):
            self.add(f"{prefix}_{i}", chunk, priority)

    def remove(self, key: str) -> None:
        self._slots = [s for s in self._slots if s.key != key]

    def clear(self, priority: Priority | None = None) -> None:
        if priority is None:
            self._slots = []
        else:
            self._slots = [s for s in self._slots if s.priority != priority]

    def build(self) -> WindowResult:
        """
        Apply priority eviction and return the final window.
        Eviction order: lowest priority first, then oldest (FIFO) within same priority.
        """
        total = sum(s.tokens for s in self._slots)
        if total <= self._effective_budget:
            return WindowResult(
                slots=list(self._slots),
                total_tokens=total,
                evicted=[],
                budget=self._effective_budget,
            )

        # sort: lowest priority first, then oldest (FIFO within priority)
        eviction_order = sorted(
            self._slots, key=lambda s: (s.priority, -s.position)
        )

        evictable = {Priority.SCRATCH, Priority.HISTORY, Priority.RETRIEVED, Priority.FEW_SHOT}
        evicted: list[ContextSlot] = []
        current_tokens = total

        for slot in eviction_order:
            if current_tokens <= self._effective_budget:
                break
            if slot.priority in evictable:
                evicted.append(slot)
                current_tokens -= slot.tokens

        if current_tokens > self._effective_budget:
            logger.warning(
                "Could not fit within budget (%d > %d); system/few-shot slots are too large",
                current_tokens, self._effective_budget,
            )

        evicted_keys = {s.key for s in evicted}
        final_slots = [s for s in self._slots if s.key not in evicted_keys]

        return WindowResult(
            slots=final_slots,
            total_tokens=current_tokens,
            evicted=evicted,
            budget=self._effective_budget,
        )

    @property
    def current_tokens(self) -> int:
        return sum(s.tokens for s in self._slots)

    @property
    def remaining_budget(self) -> int:
        return max(0, self._effective_budget - self.current_tokens)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer_name and self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            except Exception:
                pass
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)
