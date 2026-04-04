"""Minimal from-scratch RAG demo using NumPy TF-IDF retrieval."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class RetrievedChunk:
    """One retrieved chunk with similarity score."""

    chunk_id: str
    text: str
    score: float


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text with a simple alphanumeric regex."""
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 10) -> List[str]:
    """Chunk by words with overlap for context continuity."""
    words = text.split()
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    out: List[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        out.append(" ".join(words[start : start + chunk_size]))
        start += step
    return out


class MiniTfidfIndex:
    """Simple TF-IDF index with cosine similarity retrieval."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self.chunks = list(chunks)
        self.vocab: List[str] = []
        self.vocab_index: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: np.ndarray = np.array([[]], dtype=np.float64)
        self._fit()

    def _fit(self) -> None:
        tokenized = [simple_tokenize(c) for c in self.chunks]
        vocab = sorted({t for doc in tokenized for t in doc})
        self.vocab = vocab
        self.vocab_index = {t: i for i, t in enumerate(vocab)}

        n_docs = len(tokenized)
        df = np.zeros(len(vocab), dtype=np.float64)

        rows: List[np.ndarray] = []
        for tokens in tokenized:
            counts = Counter(tokens)
            row = np.zeros(len(vocab), dtype=np.float64)
            for token, count in counts.items():
                idx = self.vocab_index[token]
                row[idx] = float(count)
            rows.append(row)
            for token in set(tokens):
                df[self.vocab_index[token]] += 1.0

        tf = np.vstack(rows)
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self.matrix = self._l2_normalize(tf * self.idf)

    def embed_query(self, query: str) -> np.ndarray:
        counts = Counter(simple_tokenize(query))
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for token, count in counts.items():
            idx = self.vocab_index.get(token)
            if idx is not None:
                vec[idx] = float(count)
        return self._l2_normalize(vec * self.idf)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vec = self.embed_query(query)
        scores = self.matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk_id=f"chunk_{i}", text=self.chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            denom = float(np.linalg.norm(v)) or 1.0
            return v / denom
        denom = np.linalg.norm(v, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return v / denom


def extractive_synthesize(query: str, chunks: Iterable[RetrievedChunk]) -> str:
    """
    Return a deterministic fallback answer without external APIs.

    The heuristic picks high-overlap sentences from retrieved chunks.
    """
    query_terms = set(simple_tokenize(query))
    candidates: List[tuple[float, str]] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
            sent_terms = set(simple_tokenize(sentence))
            if not sent_terms:
                continue
            overlap = len(query_terms & sent_terms) / max(len(query_terms), 1)
            score = 0.7 * overlap + 0.3 * chunk.score
            candidates.append((score, sentence))

    if not candidates:
        return "I could not find relevant evidence in the retrieved context."

    top_sentences = [s for _, s in sorted(candidates, reverse=True)[:3]]
    return " ".join(top_sentences)


def openai_synthesize(query: str, chunks: Sequence[RetrievedChunk], model: str) -> str:
    """Use OpenAI only when API key exists and client dependency is installed."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package not installed") from exc

    context = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
    prompt = (
        "Answer using only the provided context. "
        "If evidence is insufficient, say so.\n\n"
        f"Query: {query}\n\nContext:\n{context}"
    )
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=220,
    )
    return response.output_text.strip()


def build_demo_corpus() -> List[str]:
    """Small deterministic corpus for local demos (`n=6` documents)."""
    docs = [
        (
            "Two-stage retrieval in RAG combines fast vector search with precise reranking. "
            "A bi-encoder retrieves candidate chunks quickly, and a cross-encoder reranks "
            "those candidates with richer interaction between query and passage tokens."
        ),
        (
            "RAGAS faithfulness measures whether answer claims are supported by retrieved context. "
            "Teams often track faithfulness, answer relevancy, context precision, and context recall "
            "to detect quality regressions before shipping."
        ),
        (
            "Context compression reduces token usage by removing low-information passages. "
            "A good compression strategy can lower cost while maintaining similar answer quality."
        ),
        (
            "CUPED is an A/B testing technique that reduces variance using pre-experiment covariates. "
            "Lower variance means smaller sample sizes for the same minimum detectable effect."
        ),
        (
            "O'Brien-Fleming sequential testing controls Type-I error when checking results multiple times. "
            "It sets strict early boundaries and relaxed later boundaries for interim looks."
        ),
        (
            "Circuit breakers keep multi-agent systems responsive under partial failure. "
            "After repeated errors, the breaker opens and returns degraded output instead of waiting on timeouts."
        ),
    ]
    chunks: List[str] = []
    for doc in docs:
        chunks.extend(chunk_text(doc, chunk_size=48, overlap=8))
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny from-scratch RAG demo.")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI for synthesis (requires OPENAI_API_KEY)",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="Model for OpenAI synthesis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    chunks = build_demo_corpus()
    index = MiniTfidfIndex(chunks)
    retrieved = index.retrieve(args.query, top_k=args.top_k)

    print("\n=== Retrieved Chunks ===")
    for r in retrieved:
        print(f"{r.chunk_id} | score={r.score:.3f} | {r.text[:140]}...")

    if args.use_openai:
        try:
            answer = openai_synthesize(args.query, retrieved, model=args.openai_model)
        except Exception as exc:
            print(f"\nOpenAI synthesis unavailable ({exc}). Falling back to extractive mode.")
            answer = extractive_synthesize(args.query, retrieved)
    else:
        answer = extractive_synthesize(args.query, retrieved)

    token_estimate = len(simple_tokenize(" ".join(c.text for c in retrieved)))
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Methodology ===")
    print(
        f"retrieval=TF-IDF cosine | corpus_docs=6 | total_chunks={len(chunks)} | "
        f"top_k={args.top_k} | context_token_estimate={token_estimate}"
    )


if __name__ == "__main__":
    main()

