"""
dataset_engineering/synthetic.py
----------------------------------
LLM-powered synthetic QA pair generation for RAG evaluation datasets.

Generates diverse, grounded QA pairs from retrieved documents -- covering:
  - Factoid questions
  - Multi-hop reasoning questions
  - Negative / unanswerable questions
  - Abstractive summary questions

Output is a versioned DatasetVersion ready for RAGAS evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

QuestionType = Literal["factoid", "multi_hop", "abstractive", "negative"]

_SYSTEM_PROMPT = """You are a rigorous QA dataset generator.
Given a context passage, generate high-quality question-answer pairs.
Always output valid JSON only -- no preamble, no markdown fences."""

_TEMPLATES: dict[str, str] = {
    "factoid": """Generate {n} factoid questions directly answerable from the context.
Context: {context}

Output JSON array:
[{{"question": "...", "answer": "...", "type": "factoid"}}]""",

    "multi_hop": """Generate {n} multi-hop questions requiring reasoning across multiple sentences.
Context: {context}

Output JSON array:
[{{"question": "...", "answer": "...", "type": "multi_hop", "reasoning": "..."}}]""",

    "abstractive": """Generate {n} questions requiring abstractive summarisation of the context.
Context: {context}

Output JSON array:
[{{"question": "...", "answer": "...", "type": "abstractive"}}]""",

    "negative": """Generate {n} plausible but UNANSWERABLE questions based on the context.
The question should be related to the topic but not answerable from the text.
Context: {context}

Output JSON array:
[{{"question": "...", "answer": "This cannot be answered from the provided context.", "type": "negative"}}]""",
}


@dataclass
class SyntheticConfig:
    model: str = "gpt-4o-mini"
    n_per_type: int = 2
    question_types: list[QuestionType] = field(
        default_factory=lambda: ["factoid", "multi_hop", "abstractive", "negative"]
    )
    max_context_chars: int = 2000
    temperature: float = 0.7
    max_retries: int = 3
    openai_api_key: str | None = None


# Legacy dataclasses kept for backwards compatibility
@dataclass
class SyntheticQA:
    question: str
    answer: str
    source_chunks: list[str] = field(default_factory=list)
    question_type: str = "factoid"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    samples: list[SyntheticQA] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([
            {
                "question": s.question,
                "answer": s.answer,
                "question_type": s.question_type,
                "difficulty": s.difficulty,
            }
            for s in self.samples
        ])


class SyntheticQAGenerator:
    """
    Generates synthetic QA pairs from document chunks using an LLM.

    Usage
    -----
    >>> gen = SyntheticQAGenerator()
    >>> records = gen.generate_from_chunks(chunks, n_per_chunk=3)
    >>> dv = gen.to_dataset_version(records, version="v1.0")
    """

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        self._client = None

    def generate_from_chunks(
        self,
        chunks: list[str],
        n_per_chunk: int | None = None,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Generate QA pairs from a list of document chunks.

        Returns flat list of records with keys:
          question, answer, context, type, chunk_idx
        """
        all_records = []
        n_per_type = n_per_chunk or self.config.n_per_type

        for i, chunk in enumerate(chunks):
            if show_progress:
                logger.info("Generating QA for chunk %d/%d", i + 1, len(chunks))
            records = self._generate_for_chunk(chunk, i, n_per_type)
            all_records.extend(records)

        logger.info("Generated %d QA pairs from %d chunks", len(all_records), len(chunks))
        return all_records

    def to_dataset_version(
        self,
        records: list[dict],
        version: str = "v1.0",
        dataset_name: str = "synthetic_qa",
    ):
        """Convert generated records to a DatasetVersion."""
        import pandas as pd
        from dataset_engineering.versioning import DatasetLineage, DatasetVersion

        df = pd.DataFrame(records)
        lineage = DatasetLineage(
            source_hash="synthetic",
            transform="SyntheticQAGenerator",
            params={
                "model": self.config.model,
                "n_per_type": self.config.n_per_type,
                "question_types": self.config.question_types,
            },
        )
        return DatasetVersion(
            data=df,
            version=version,
            dataset_name=dataset_name,
            lineage=lineage,
        )

    def augment_with_paraphrases(
        self,
        records: list[dict],
        question_col: str = "question",
        n_paraphrases: int = 2,
    ) -> list[dict]:
        """
        Augment dataset with paraphrased questions for robustness testing.
        Each original record produces n_paraphrases additional variants.
        """
        augmented = list(records)
        for rec in records:
            paraphrases = self._paraphrase(rec[question_col], n=n_paraphrases)
            for para in paraphrases:
                new_rec = dict(rec)
                new_rec[question_col] = para
                new_rec["is_paraphrase"] = True
                augmented.append(new_rec)
        return augmented

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_for_chunk(
        self, chunk: str, chunk_idx: int, n_per_type: int
    ) -> list[dict]:
        records = []
        context = chunk[: self.config.max_context_chars]

        for q_type in self.config.question_types:
            prompt = _TEMPLATES[q_type].format(context=context, n=n_per_type)
            raw = self._call_llm(prompt)
            parsed = self._parse_response(raw)
            for item in parsed:
                records.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "context": context,
                    "type": q_type,
                    "chunk_idx": chunk_idx,
                    "reasoning": item.get("reasoning", ""),
                })
        return records

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with exponential backoff retry."""
        client = self._get_client()
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d): %s. Retrying in %ds",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries")

    def _parse_response(self, raw: str) -> list[dict]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            # Handle {items: [...]} or similar wrapper
            for key in ["items", "questions", "pairs", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return []

    def _paraphrase(self, question: str, n: int) -> list[str]:
        prompt = (
            f"Generate {n} paraphrases of this question. "
            f"Output JSON array of strings only.\n"
            f"Question: {question}"
        )
        raw = self._call_llm(prompt)
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result[:n]
        except Exception:
            pass
        return []

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {}
                if self.config.openai_api_key:
                    kwargs["api_key"] = self.config.openai_api_key
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
