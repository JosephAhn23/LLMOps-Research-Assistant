"""
context_engineering/few_shot.py
---------------------------------
Dynamic few-shot selection: retrieve semantically similar examples
from an example store at inference time rather than hardcoding them.

Backed by FAISS for fast similarity search over example embeddings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Example:
    id: str
    input: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def to_prompt_str(self, input_label: str = "Q", output_label: str = "A") -> str:
        return f"{input_label}: {self.input}\n{output_label}: {self.output}"


# Alias for backwards compatibility with __init__.py export
FewShotExample = Example


class DynamicFewShot:
    """
    Retrieves semantically relevant few-shot examples for a given query.

    Usage
    -----
    >>> store = DynamicFewShot(encoder="sentence-transformers/all-MiniLM-L6-v2")
    >>> store.add_examples(examples)
    >>> shots = store.retrieve(query="What is RLHF?", k=3)
    >>> prompt = store.build_prompt(query, shots)
    """

    def __init__(
        self,
        encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity: str = "cosine",
    ):
        self.encoder_name = encoder
        self.similarity = similarity
        self._examples: list[Example] = []
        self._index = None
        self._encoder = None

    # ------------------------------------------------------------------
    # Building the store
    # ------------------------------------------------------------------

    def add_examples(self, examples: list[Example], batch_size: int = 64) -> None:
        """Encode examples and add them to the FAISS index."""
        encoder = self._get_encoder()
        texts = [ex.input for ex in examples]
        embeddings = self._encode_batch(encoder, texts, batch_size)

        for ex, emb in zip(examples, embeddings):
            ex.embedding = emb
        self._examples.extend(examples)
        self._build_index()
        logger.info("DynamicFewShot store: %d examples", len(self._examples))

    def load_from_jsonl(self, path: str | Path) -> None:
        """Load examples from a JSONL file with keys: id, input, output, metadata."""
        path = Path(path)
        examples = []
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                examples.append(Example(**obj))
        self.add_examples(examples)

    def save_to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w") as f:
            for ex in self._examples:
                obj = {
                    "id": ex.id,
                    "input": ex.input,
                    "output": ex.output,
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(obj) + "\n")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 3, diversity: bool = True) -> list[Example]:
        """
        Retrieve top-k semantically similar examples.

        Parameters
        ----------
        query     : the current user query
        k         : number of examples to return
        diversity : apply MMR (Maximal Marginal Relevance) for diversity
        """
        if not self._examples:
            logger.warning("No examples in store")
            return []

        encoder = self._get_encoder()
        query_emb = self._encode_batch(encoder, [query], batch_size=1)[0]

        if diversity and len(self._examples) > k:
            return self._mmr_retrieve(query_emb, k)
        else:
            return self._faiss_retrieve(query_emb, k)

    def build_prompt(
        self,
        query: str,
        examples: list[Example],
        system_prompt: str = "",
        input_label: str = "Q",
        output_label: str = "A",
    ) -> str:
        """
        Build a few-shot prompt string from retrieved examples.

        Returns a formatted prompt ready for the LLM.
        """
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
            parts.append("")

        for ex in examples:
            parts.append(ex.to_prompt_str(input_label, output_label))
            parts.append("")

        parts.append(f"{input_label}: {query}")
        parts.append(f"{output_label}:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _faiss_retrieve(self, query_emb: np.ndarray, k: int) -> list[Example]:
        import faiss
        k = min(k, len(self._examples))
        D, I = self._index.search(query_emb.reshape(1, -1).astype("float32"), k)
        return [self._examples[i] for i in I[0] if i >= 0]

    def _mmr_retrieve(
        self, query_emb: np.ndarray, k: int, lambda_: float = 0.5
    ) -> list[Example]:
        """Maximal Marginal Relevance for diverse yet relevant examples."""
        all_embs = np.stack([ex.embedding for ex in self._examples]).astype("float32")
        query_sims = self._cosine_sim(query_emb, all_embs)

        selected_idx: list[int] = []
        remaining = list(range(len(self._examples)))

        for _ in range(min(k, len(self._examples))):
            if not selected_idx:
                best = int(np.argmax(query_sims))
            else:
                sel_embs = all_embs[selected_idx]
                redundancy = self._cosine_sim_matrix(
                    all_embs[remaining], sel_embs
                ).max(axis=1)
                relevance = query_sims[remaining]
                mmr_scores = lambda_ * relevance - (1 - lambda_) * redundancy
                best = remaining[int(np.argmax(mmr_scores))]

            selected_idx.append(best)
            remaining = [i for i in remaining if i != best]

        return [self._examples[i] for i in selected_idx]

    def _build_index(self) -> None:
        import faiss
        embs = np.stack([ex.embedding for ex in self._examples]).astype("float32")
        if self.similarity == "cosine":
            faiss.normalize_L2(embs)
        dim = embs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embs)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _encode_batch(
        self, encoder: Any, texts: list[str], batch_size: int
    ) -> np.ndarray:
        return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    @staticmethod
    def _cosine_sim(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-9)
        return (matrix / norms) @ (vec / np.linalg.norm(vec).clip(1e-9))

    @staticmethod
    def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-9)
        B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-9)
        return A @ B.T
