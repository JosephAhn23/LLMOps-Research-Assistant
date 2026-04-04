"""
context_engineering/chain_of_thought.py
-----------------------------------------
Structured Chain-of-Thought (CoT) prompt construction.

Supports:
  - Zero-shot CoT ("Let's think step by step")
  - Few-shot CoT (examples with reasoning traces)
  - Self-consistency prompting (multiple reasoning paths)
  - Tree-of-Thought (ToT) branching prompts
  - Scratchpad prompting for retrieval-augmented reasoning
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CoTExample:
    question: str
    reasoning: str
    answer: str

    def format(self, numbered: bool = False, step_prefix: str = "Step") -> str:
        steps = [s.strip() for s in self.reasoning.split("\n") if s.strip()]
        if numbered:
            reasoning_str = "\n".join(
                f"{step_prefix} {i + 1}: {s}" for i, s in enumerate(steps)
            )
        else:
            reasoning_str = "\n".join(steps)
        return f"Q: {self.question}\nReasoning:\n{reasoning_str}\nA: {self.answer}"


@dataclass
class ChainOfThoughtConfig:
    strategy: Literal["zero_shot", "few_shot", "self_consistency", "tot", "scratchpad"] = "few_shot"
    examples: list[CoTExample] = field(default_factory=list)
    n_paths: int = 3                 # for self_consistency / tot
    numbered_steps: bool = True
    step_prefix: str = "Step"
    system_instruction: str = (
        "You are a precise and rigorous reasoning assistant. "
        "Always show your work before giving the final answer."
    )


# Legacy alias so existing imports of CoTResult don't break
@dataclass
class CoTResult:
    strategy: str
    prompt: str

    def summary(self) -> str:
        return f"CoT ({self.strategy}): {len(self.prompt)} chars"


class ChainOfThoughtBuilder:
    """
    Builds Chain-of-Thought prompts for the LLMOps pipeline.

    Example
    -------
    >>> builder = ChainOfThoughtBuilder()
    >>> prompt = builder.build(
    ...     query="Why does FAISS use IVF indexing?",
    ...     context="FAISS supports exact and approximate search...",
    ...     strategy="zero_shot",
    ... )
    """

    def __init__(self, config: ChainOfThoughtConfig | None = None):
        self.config = config or ChainOfThoughtConfig()

    def build(
        self,
        query: str,
        context: str = "",
        strategy: str | None = None,
    ) -> str:
        strategy = strategy or self.config.strategy
        builders = {
            "zero_shot": self._zero_shot,
            "few_shot": self._few_shot,
            "self_consistency": self._self_consistency,
            "tot": self._tree_of_thought,
            "scratchpad": self._scratchpad,
        }
        if strategy not in builders:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(builders)}")
        return builders[strategy](query, context)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _zero_shot(self, query: str, context: str) -> str:
        parts = [self.config.system_instruction]
        if context:
            parts.append(f"\nContext:\n{context}")
        parts.append(
            f"\nQuestion: {query}\n\nLet's think through this step by step:"
        )
        return "\n".join(parts)

    def _few_shot(self, query: str, context: str) -> str:
        parts = [self.config.system_instruction, ""]
        for ex in self.config.examples:
            parts.append(ex.format(
                numbered=self.config.numbered_steps,
                step_prefix=self.config.step_prefix,
            ))
            parts.append("")

        if context:
            parts.append(f"Context:\n{context}\n")
        parts.append(f"Q: {query}\nReasoning:")
        return "\n".join(parts)

    def _self_consistency(self, query: str, context: str) -> str:
        """
        Prompts the model to generate N independent reasoning paths.
        The caller should call the LLM n_paths times and take majority vote.
        """
        base = self._zero_shot(query, context)
        suffix = textwrap.dedent(f"""

            Generate {self.config.n_paths} independent reasoning paths to this question,
            then choose the most consistent answer across paths.
            Format each path as:
            Path 1: <reasoning> -> Answer: <answer>
            Path 2: <reasoning> -> Answer: <answer>
            ...
            Final Answer (most consistent): <answer>
        """)
        return base + suffix

    def _tree_of_thought(self, query: str, context: str) -> str:
        """Tree-of-Thought: branch -> evaluate -> select best branch."""
        parts = [
            self.config.system_instruction,
            "",
            "Use Tree-of-Thought reasoning:",
            f"  1. Generate {self.config.n_paths} possible approaches to the problem.",
            "  2. Evaluate each approach for correctness and completeness.",
            "  3. Select the best approach and elaborate it fully.",
            "",
        ]
        if context:
            parts.append(f"Context:\n{context}\n")
        parts.append(f"Question: {query}\n")
        parts.append("Branches:")
        for i in range(1, self.config.n_paths + 1):
            parts.append(f"  Branch {i}: [explore approach {i}]")
        parts.append("\nEvaluation:")
        parts.append("Best Branch: [select and elaborate]")
        return "\n".join(parts)

    def _scratchpad(self, query: str, context: str) -> str:
        """
        Scratchpad prompting for retrieval-augmented reasoning.
        The model writes intermediate notes before the final answer.
        """
        parts = [
            self.config.system_instruction,
            "",
            "You have access to retrieved context. Use a scratchpad to:",
            "  1. Note relevant facts from the context",
            "  2. Identify gaps or contradictions",
            "  3. Reason through the question",
            "  4. State your final answer",
            "",
        ]
        if context:
            parts.append(f"<context>\n{context}\n</context>\n")
        parts.append(f"<question>{query}</question>\n")
        parts.append("<scratchpad>")
        parts.append("Relevant facts:")
        parts.append("Gaps/contradictions:")
        parts.append("Reasoning:")
        parts.append("</scratchpad>")
        parts.append("<answer>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    @classmethod
    def rag_scratchpad(cls, query: str, chunks: list[str]) -> str:
        """Build a RAG-optimised scratchpad prompt from retrieved chunks."""
        context = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )
        builder = cls(ChainOfThoughtConfig(strategy="scratchpad"))
        return builder.build(query, context=context)

    @classmethod
    def zero_shot_cot(cls, query: str, context: str = "") -> str:
        return cls().build(query, context=context, strategy="zero_shot")
