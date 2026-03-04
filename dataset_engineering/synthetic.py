"""
Synthetic QA dataset generation for RAG fine-tuning and evaluation.

Generates:
  - Factoid QA pairs from source documents
  - Multi-hop reasoning questions requiring multiple document chunks
  - Negative (unanswerable) examples for robustness training
  - Adversarial questions designed to test retrieval precision

All generation uses an LLM via a configurable callable. Falls back to
template-based generation if no LLM is provided (useful for CI/testing).

Usage:
    generator = SyntheticDataGenerator(llm_fn=my_openai_fn)
    dataset = generator.generate(
        documents=chunks,
        n_factoid=100,
        n_multihop=50,
        n_negative=30,
    )
    df = dataset.to_dataframe()
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]


@dataclass
class SyntheticQA:
    question: str
    answer: str
    source_chunks: list[str]
    question_type: str  # "factoid" | "multihop" | "negative" | "adversarial"
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    samples: list[SyntheticQA]

    def to_dataframe(self) -> "Any":
        import pandas as pd
        return pd.DataFrame([
            {
                "question": s.question,
                "answer": s.answer,
                "source_chunks": " | ".join(s.source_chunks),
                "question_type": s.question_type,
                "difficulty": s.difficulty,
            }
            for s in self.samples
        ])

    def filter_by_type(self, question_type: str) -> "SyntheticDataset":
        return SyntheticDataset([s for s in self.samples if s.question_type == question_type])

    def summary(self) -> str:
        from collections import Counter
        type_counts = Counter(s.question_type for s in self.samples)
        diff_counts = Counter(s.difficulty for s in self.samples)
        lines = [
            f"SyntheticDataset: {len(self.samples)} samples",
            f"  Types: {dict(type_counts)}",
            f"  Difficulty: {dict(diff_counts)}",
        ]
        return "\n".join(lines)


# ── Prompt templates ──────────────────────────────────────────────────────────

FACTOID_PROMPT = """Based on the following document excerpt, generate a factoid question and its answer.
The question should be answerable directly from the text.

Document:
{document}

Generate a JSON object with keys "question" and "answer":
"""

MULTIHOP_PROMPT = """Based on the following two document excerpts, generate a multi-hop question that requires
information from BOTH documents to answer correctly.

Document 1:
{doc1}

Document 2:
{doc2}

Generate a JSON object with keys "question" and "answer":
"""

NEGATIVE_PROMPT = """Based on the following document excerpt, generate a question that CANNOT be answered
from the document (the answer is not in the text). The question should be plausible and related to the topic.

Document:
{document}

Generate a JSON object with keys "question" and "answer" where answer is "unanswerable":
"""

ADVERSARIAL_PROMPT = """Based on the following document excerpt, generate an adversarial question designed
to test retrieval precision. The question should use similar vocabulary to the document but ask about
something subtly different or require careful reading.

Document:
{document}

Generate a JSON object with keys "question" and "answer":
"""


def _parse_llm_qa(response: str) -> tuple[str, str]:
    """Extract question/answer from LLM JSON response."""
    import json
    try:
        # Try to find JSON block
        match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            return obj.get("question", ""), obj.get("answer", "")
    except Exception:
        pass

    # Fallback: regex extraction
    q_match = re.search(r'"question"\s*:\s*"([^"]+)"', response)
    a_match = re.search(r'"answer"\s*:\s*"([^"]+)"', response)
    question = q_match.group(1) if q_match else ""
    answer = a_match.group(1) if a_match else ""
    return question, answer


def _template_factoid(document: str) -> tuple[str, str]:
    """Template-based factoid generation (no LLM required)."""
    sentences = [s.strip() for s in document.split(".") if len(s.strip()) > 20]
    if not sentences:
        return "What is the main topic?", "See the document."
    sentence = random.choice(sentences[:5])
    words = sentence.split()
    if len(words) > 3:
        # Replace a key noun with "what"
        question = f"What {' '.join(words[1:min(8, len(words))])}?"
        return question, sentence
    return "What is described in this passage?", sentence


class SyntheticQAGenerator:
    """
    Generates synthetic QA datasets for RAG training and evaluation.

    Parameters
    ----------
    llm_fn : LLMFn | None
        Callable that takes a prompt and returns a completion. If None,
        uses template-based generation (no API key required).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        llm_fn: LLMFn | None = None,
        seed: int = 42,
    ) -> None:
        self.llm_fn = llm_fn
        random.seed(seed)

    def generate(
        self,
        documents: list[str],
        n_factoid: int = 50,
        n_multihop: int = 20,
        n_negative: int = 15,
        n_adversarial: int = 10,
    ) -> SyntheticDataset:
        """
        Generate a mixed synthetic dataset.

        Parameters
        ----------
        documents : list[str]
            Source document chunks to generate questions from.
        n_factoid : int
            Number of factoid QA pairs.
        n_multihop : int
            Number of multi-hop questions (requires >= 2 documents).
        n_negative : int
            Number of unanswerable questions.
        n_adversarial : int
            Number of adversarial questions.
        """
        samples: list[SyntheticQA] = []

        samples.extend(self._generate_factoid(documents, n_factoid))
        if len(documents) >= 2:
            samples.extend(self._generate_multihop(documents, n_multihop))
        samples.extend(self._generate_negative(documents, n_negative))
        samples.extend(self._generate_adversarial(documents, n_adversarial))

        logger.info(
            "Generated %d synthetic QA pairs (%d factoid, %d multihop, %d negative, %d adversarial)",
            len(samples), n_factoid, n_multihop, n_negative, n_adversarial,
        )
        return SyntheticDataset(samples)

    def _generate_factoid(self, documents: list[str], n: int) -> list[SyntheticQA]:
        samples = []
        docs = random.choices(documents, k=n)
        for doc in docs:
            q, a = self._qa_from_prompt(FACTOID_PROMPT.format(document=doc[:1000]), doc)
            if q:
                samples.append(SyntheticQA(
                    question=q, answer=a,
                    source_chunks=[doc],
                    question_type="factoid",
                    difficulty="easy",
                ))
        return samples

    def _generate_multihop(self, documents: list[str], n: int) -> list[SyntheticQA]:
        samples = []
        for _ in range(n):
            doc1, doc2 = random.sample(documents, 2)
            prompt = MULTIHOP_PROMPT.format(doc1=doc1[:500], doc2=doc2[:500])
            q, a = self._qa_from_prompt(prompt, doc1 + " " + doc2)
            if q:
                samples.append(SyntheticQA(
                    question=q, answer=a,
                    source_chunks=[doc1, doc2],
                    question_type="multihop",
                    difficulty="hard",
                ))
        return samples

    def _generate_negative(self, documents: list[str], n: int) -> list[SyntheticQA]:
        samples = []
        docs = random.choices(documents, k=n)
        for doc in docs:
            q, _ = self._qa_from_prompt(NEGATIVE_PROMPT.format(document=doc[:1000]), doc)
            if q:
                samples.append(SyntheticQA(
                    question=q, answer="unanswerable",
                    source_chunks=[doc],
                    question_type="negative",
                    difficulty="medium",
                ))
        return samples

    def _generate_adversarial(self, documents: list[str], n: int) -> list[SyntheticQA]:
        samples = []
        docs = random.choices(documents, k=n)
        for doc in docs:
            q, a = self._qa_from_prompt(ADVERSARIAL_PROMPT.format(document=doc[:1000]), doc)
            if q:
                samples.append(SyntheticQA(
                    question=q, answer=a,
                    source_chunks=[doc],
                    question_type="adversarial",
                    difficulty="hard",
                ))
        return samples

    def _qa_from_prompt(self, prompt: str, fallback_doc: str) -> tuple[str, str]:
        if self.llm_fn is not None:
            try:
                response = self.llm_fn(prompt)
                q, a = _parse_llm_qa(response)
                if q and a:
                    return q, a
            except Exception as e:
                logger.debug("LLM generation failed: %s", e)
        return _template_factoid(fallback_doc)
