"""
Chain-of-thought prompting strategies.

Implements:
  - Zero-shot CoT: "Let's think step by step."
  - Few-shot CoT: examples with explicit reasoning chains
  - Self-consistency: sample N reasoning paths, majority-vote the answer
  - Tree-of-Thought (ToT): breadth-first search over reasoning branches
  - Scratchpad: structured intermediate computation space

Usage:
    cot = ChainOfThought(strategy="self_consistency", n_samples=5)
    result = cot.run(query, context, llm_fn=my_llm)
    print(result.answer, result.confidence)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]


@dataclass
class CoTResult:
    strategy: str
    query: str
    answer: str
    reasoning: str
    confidence: float
    all_samples: list[str] = field(default_factory=list)
    token_cost: int = 0

    def summary(self) -> str:
        return (
            f"CoT ({self.strategy})\n"
            f"  Answer:     {self.answer[:100]}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Reasoning:  {self.reasoning[:200]}...\n"
            f"  Samples:    {len(self.all_samples)}"
        )


# ── Prompt templates ──────────────────────────────────────────────────────────

ZERO_SHOT_COT_TEMPLATE = """{context}

Question: {query}

Let's think step by step.
"""

FEW_SHOT_COT_TEMPLATE = """{examples}

Context: {context}

Question: {query}

Let's think step by step.
"""

SELF_CONSISTENCY_TEMPLATE = """{context}

Question: {query}

Think through this carefully, then give your final answer on a line starting with "Answer:".
"""

TOT_BRANCH_TEMPLATE = """Context: {context}

Question: {query}

Reasoning branch {branch_id}: {branch_hint}

Continue this reasoning and provide a partial answer:
"""

SCRATCHPAD_TEMPLATE = """Context: {context}

Question: {query}

Use the scratchpad below to work through the problem step by step.
<scratchpad>
{scratchpad}
</scratchpad>

Based on your scratchpad, provide the final answer:
"""


def _extract_answer(text: str) -> str:
    """Extract the answer from a CoT response."""
    # Look for "Answer: ..." pattern
    match = re.search(r"(?:Answer|Therefore|Thus|So)[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fall back to last sentence
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences[-1] if sentences else text.strip()


def _majority_vote(answers: list[str]) -> tuple[str, float]:
    """Return the most common answer and its fraction."""
    if not answers:
        return "", 0.0
    counts = Counter(answers)
    best, count = counts.most_common(1)[0]
    return best, count / len(answers)


class ChainOfThoughtBuilder:
    """
    Chain-of-thought prompting with multiple strategies.

    Parameters
    ----------
    strategy : str
        "zero_shot" | "few_shot" | "self_consistency" | "tree_of_thought" | "scratchpad"
    n_samples : int
        Number of samples for self-consistency (ignored for other strategies).
    n_branches : int
        Number of branches for tree-of-thought.
    examples : list[dict] | None
        Few-shot examples with keys "question", "reasoning", "answer".
    """

    def __init__(
        self,
        strategy: Literal[
            "zero_shot", "few_shot", "self_consistency", "tree_of_thought", "scratchpad"
        ] = "zero_shot",
        n_samples: int = 5,
        n_branches: int = 3,
        examples: list[dict] | None = None,
    ) -> None:
        self.strategy = strategy
        self.n_samples = n_samples
        self.n_branches = n_branches
        self.examples = examples or []

    def run(
        self,
        query: str,
        context: str = "",
        llm_fn: LLMFn | None = None,
    ) -> CoTResult:
        """
        Run the selected CoT strategy.

        Parameters
        ----------
        query : str
            The user question.
        context : str
            Retrieved context to ground the answer.
        llm_fn : callable | None
            Function that takes a prompt string and returns a completion string.
            If None, returns a mock result (for testing).
        """
        if llm_fn is None:
            llm_fn = self._mock_llm

        dispatch = {
            "zero_shot": self._zero_shot,
            "few_shot": self._few_shot,
            "self_consistency": self._self_consistency,
            "tree_of_thought": self._tree_of_thought,
            "scratchpad": self._scratchpad,
        }
        fn = dispatch.get(self.strategy, self._zero_shot)
        return fn(query, context, llm_fn)

    def _zero_shot(self, query: str, context: str, llm_fn: LLMFn) -> CoTResult:
        prompt = ZERO_SHOT_COT_TEMPLATE.format(context=context, query=query)
        response = llm_fn(prompt)
        answer = _extract_answer(response)
        return CoTResult(
            strategy="zero_shot",
            query=query,
            answer=answer,
            reasoning=response,
            confidence=0.7,
            all_samples=[response],
            token_cost=len(prompt.split()) + len(response.split()),
        )

    def _few_shot(self, query: str, context: str, llm_fn: LLMFn) -> CoTResult:
        examples_str = "\n\n".join(
            f"Q: {e['question']}\nReasoning: {e.get('reasoning', '')}\nA: {e['answer']}"
            for e in self.examples
        )
        prompt = FEW_SHOT_COT_TEMPLATE.format(
            examples=examples_str, context=context, query=query
        )
        response = llm_fn(prompt)
        answer = _extract_answer(response)
        return CoTResult(
            strategy="few_shot",
            query=query,
            answer=answer,
            reasoning=response,
            confidence=0.8,
            all_samples=[response],
            token_cost=len(prompt.split()) + len(response.split()),
        )

    def _self_consistency(self, query: str, context: str, llm_fn: LLMFn) -> CoTResult:
        """
        Sample N independent reasoning paths, then majority-vote the final answer.
        Reduces variance from single-sample CoT.
        """
        prompt = SELF_CONSISTENCY_TEMPLATE.format(context=context, query=query)
        responses = [llm_fn(prompt) for _ in range(self.n_samples)]
        answers = [_extract_answer(r) for r in responses]
        best_answer, confidence = _majority_vote(answers)

        logger.debug(
            "Self-consistency: %d samples, majority answer confidence=%.2f",
            self.n_samples, confidence,
        )

        return CoTResult(
            strategy="self_consistency",
            query=query,
            answer=best_answer,
            reasoning=responses[0],
            confidence=confidence,
            all_samples=responses,
            token_cost=self.n_samples * (len(prompt.split()) + 50),
        )

    def _tree_of_thought(self, query: str, context: str, llm_fn: LLMFn) -> CoTResult:
        """
        Breadth-first search over reasoning branches.
        Each branch explores a different reasoning direction; the best-scoring
        branch (by answer consistency) is selected.
        """
        branch_hints = [
            "Focus on the factual claims in the context.",
            "Consider what information might be missing.",
            "Think about potential counterarguments.",
        ][: self.n_branches]

        branch_responses = []
        for i, hint in enumerate(branch_hints):
            prompt = TOT_BRANCH_TEMPLATE.format(
                context=context, query=query, branch_id=i + 1, branch_hint=hint
            )
            branch_responses.append(llm_fn(prompt))

        answers = [_extract_answer(r) for r in branch_responses]
        best_answer, confidence = _majority_vote(answers)

        # Select the branch whose answer matches the majority
        best_reasoning = branch_responses[0]
        for r, a in zip(branch_responses, answers):
            if a == best_answer:
                best_reasoning = r
                break

        return CoTResult(
            strategy="tree_of_thought",
            query=query,
            answer=best_answer,
            reasoning=best_reasoning,
            confidence=confidence,
            all_samples=branch_responses,
            token_cost=self.n_branches * 100,
        )

    def _scratchpad(self, query: str, context: str, llm_fn: LLMFn) -> CoTResult:
        """
        Two-pass: first generate a scratchpad, then use it to produce the answer.
        Useful for multi-step arithmetic or structured reasoning.
        """
        # Pass 1: generate scratchpad
        scratchpad_prompt = (
            f"Context: {context}\n\nQuestion: {query}\n\n"
            "Write a step-by-step scratchpad working through the problem:"
        )
        scratchpad = llm_fn(scratchpad_prompt)

        # Pass 2: use scratchpad to answer
        answer_prompt = SCRATCHPAD_TEMPLATE.format(
            context=context, query=query, scratchpad=scratchpad
        )
        response = llm_fn(answer_prompt)
        answer = _extract_answer(response)

        return CoTResult(
            strategy="scratchpad",
            query=query,
            answer=answer,
            reasoning=f"Scratchpad:\n{scratchpad}\n\nFinal:\n{response}",
            confidence=0.75,
            all_samples=[scratchpad, response],
            token_cost=len(scratchpad_prompt.split()) + len(answer_prompt.split()) + 100,
        )

    @staticmethod
    def _mock_llm(prompt: str) -> str:
        """Mock LLM for testing without an API key."""
        return (
            "Step 1: Analyze the question.\n"
            "Step 2: Review the context.\n"
            "Step 3: Synthesize an answer.\n"
            "Answer: This is a mock response for testing."
        )
