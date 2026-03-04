"""
Dynamic few-shot example selection.

Retrieves the most relevant few-shot examples for a given query using
FAISS embedding similarity, then applies MMR (Maximal Marginal Relevance)
to ensure diversity -- avoiding redundant examples that all cover the same
reasoning pattern.

Falls back to random selection if sentence-transformers is not available.

Usage:
    selector = FewShotSelector(examples=EXAMPLE_POOL)
    selected = selector.select(query="What is RAG?", k=3)
    prompt = selector.build_prompt(query, selected)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    question: str
    answer: str
    metadata: dict[str, Any] | None = None

    def to_prompt_str(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with sentence-transformers; fall back to TF-IDF vectors."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except ImportError:
        logger.debug("sentence-transformers not available; using TF-IDF fallback")
        return _tfidf_vectors(texts)


def _tfidf_vectors(texts: list[str]) -> np.ndarray:
    """Sparse TF-IDF vectors normalised to unit length."""
    import re
    from collections import Counter

    def tokenize(t: str) -> list[str]:
        return re.findall(r"\b\w+\b", t.lower())

    tokenized = [tokenize(t) for t in texts]
    vocab = sorted({tok for toks in tokenized for tok in toks})
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    vecs = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        for tok, cnt in Counter(toks).items():
            if tok in vocab_idx:
                vecs[i, vocab_idx[tok]] = cnt
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-9)


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """
    Maximal Marginal Relevance selection.

    Balances relevance to the query (lambda_) against diversity among
    selected examples (1 - lambda_).

    Returns indices into candidate_vecs.
    """
    selected: list[int] = []
    remaining = list(range(len(candidate_vecs)))

    for _ in range(min(k, len(candidate_vecs))):
        if not remaining:
            break

        relevance = candidate_vecs[remaining] @ query_vec

        if not selected:
            best_local = int(np.argmax(relevance))
        else:
            selected_vecs = candidate_vecs[selected]
            # Max similarity to any already-selected example
            redundancy = (candidate_vecs[remaining] @ selected_vecs.T).max(axis=1)
            mmr_scores = lambda_ * relevance - (1 - lambda_) * redundancy
            best_local = int(np.argmax(mmr_scores))

        selected.append(remaining[best_local])
        remaining.pop(best_local)

    return selected


class DynamicFewShot:
    """
    FAISS-backed dynamic few-shot selector with MMR diversity.

    Parameters
    ----------
    examples : list[FewShotExample]
        Pool of candidate examples to select from.
    lambda_ : float
        MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    """

    def __init__(
        self,
        examples: list[FewShotExample],
        lambda_: float = 0.7,
    ) -> None:
        self.examples = examples
        self.lambda_ = lambda_
        self._vecs: np.ndarray | None = None
        self._index: Any | None = None
        self._build_index()

    def _build_index(self) -> None:
        if not self.examples:
            return
        texts = [f"{e.question} {e.answer}" for e in self.examples]
        self._vecs = _embed_texts(texts)

        try:
            import faiss
            dim = self._vecs.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._vecs.astype(np.float32))
            logger.debug("FewShotSelector: FAISS index built (%d examples)", len(self.examples))
        except ImportError:
            logger.debug("faiss not available; using numpy dot-product search")

    def select(self, query: str, k: int = 3) -> list[FewShotExample]:
        """
        Select k diverse, relevant examples for the given query.

        Uses FAISS for fast candidate retrieval, then MMR for diversity.
        """
        if not self.examples:
            return []

        if self._vecs is None:
            return random.sample(self.examples, min(k, len(self.examples)))

        query_vec = _embed_texts([query])[0]

        # Retrieve top-2k candidates by relevance
        n_candidates = min(k * 2, len(self.examples))
        if self._index is not None:
            _, candidate_ids = self._index.search(
                query_vec.reshape(1, -1).astype(np.float32), n_candidates
            )
            candidate_ids = candidate_ids[0].tolist()
        else:
            sims = self._vecs @ query_vec
            candidate_ids = list(np.argsort(-sims)[:n_candidates])

        candidate_vecs = self._vecs[candidate_ids]
        mmr_local = _mmr(query_vec, candidate_vecs, k=k, lambda_=self.lambda_)
        selected_ids = [candidate_ids[i] for i in mmr_local]

        return [self.examples[i] for i in selected_ids]

    def build_prompt(
        self,
        query: str,
        examples: list[FewShotExample] | None = None,
        k: int = 3,
        system_prefix: str = "",
    ) -> str:
        """
        Build a complete few-shot prompt string.

        Parameters
        ----------
        query : str
            The current user query.
        examples : list[FewShotExample] | None
            Pre-selected examples; if None, calls select(query, k).
        k : int
            Number of examples to select if examples is None.
        system_prefix : str
            Optional system instruction prepended to the prompt.
        """
        if examples is None:
            examples = self.select(query, k=k)

        parts = []
        if system_prefix:
            parts.append(system_prefix.strip())
            parts.append("")

        for ex in examples:
            parts.append(ex.to_prompt_str())
            parts.append("")

        parts.append(f"Q: {query}")
        parts.append("A:")

        return "\n".join(parts)

    def add_example(self, example: FewShotExample) -> None:
        """Add a new example to the pool and rebuild the index."""
        self.examples.append(example)
        self._build_index()
