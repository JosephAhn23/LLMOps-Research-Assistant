"""
context_engineering/compressor.py
-----------------------------------
Prompt compression via token-level perplexity pruning (LLMLingua-style).

Ranks each sentence by its perplexity under a small proxy LM (GPT-2 by default).
Low-perplexity sentences are "predictable" and can be pruned without information loss.
High-perplexity sentences are surprising/informative and should be kept.

Reference: LLMLingua: Compressing Prompts for Accelerated Inference of LLMs (EMNLP 2023)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    def summary(self) -> str:
        return (
            f"Tokens: {self.original_tokens} -> {self.compressed_tokens} "
            f"({self.compression_ratio:.1%} ratio, {self.tokens_saved} saved)"
        )


class PromptCompressor:
    """
    Compresses retrieved context chunks to fit within a token budget.

    Strategies
    ----------
    perplexity  : Keep high-perplexity (informative) sentences. Requires transformers.
    extractive  : Simple TF-IDF sentence ranking (no GPU needed).
    truncate    : Naive tail truncation (baseline).

    Example
    -------
    >>> compressor = PromptCompressor(target_ratio=0.5)
    >>> result = compressor.compress(long_context, query="What is FAISS?")
    >>> print(result.summary())
    """

    def __init__(
        self,
        target_ratio: float = 0.5,
        strategy: Literal["perplexity", "extractive", "truncate"] = "extractive",
        proxy_model: str = "gpt2",
        min_sentences: int = 3,
    ):
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        self.target_ratio = target_ratio
        self.strategy = strategy
        self.proxy_model = proxy_model
        self.min_sentences = min_sentences
        self._model = None
        self._tokenizer = None

    def compress(self, text: str, query: str = "") -> CompressionResult:
        """
        Compress text to approximately target_ratio of its original token count.

        Parameters
        ----------
        text  : context to compress
        query : optional query for relevance-aware compression
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_sentences:
            tokens = self._count_tokens(text)
            return CompressionResult(text, text, tokens, tokens)

        scores = self._score_sentences(sentences, query)
        compressed = self._select_sentences(sentences, scores)

        orig_tokens = self._count_tokens(text)
        comp_tokens = self._count_tokens(compressed)

        logger.info(
            "Compressed %d -> %d tokens (%.1f%%)",
            orig_tokens, comp_tokens, 100 * comp_tokens / max(orig_tokens, 1),
        )

        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
        )

    def compress_chunks(
        self, chunks: list[str], query: str = ""
    ) -> list[CompressionResult]:
        """Compress a list of retrieved chunks independently."""
        return [self.compress(chunk, query) for chunk in chunks]

    def fit_to_budget(
        self, chunks: list[str], token_budget: int, query: str = ""
    ) -> list[str]:
        """
        Compress and select chunks to fit within a token budget.
        Prioritises chunks by their relevance score.
        """
        results = self.compress_chunks(chunks, query)
        selected, total = [], 0
        for r in sorted(results, key=lambda x: x.compression_ratio):
            if total + r.compressed_tokens <= token_budget:
                selected.append(r.compressed_text)
                total += r.compressed_tokens
            else:
                remaining = token_budget - total
                if remaining > 50:
                    # truncate to fit remaining budget (rough char estimate)
                    partial = r.compressed_text[: remaining * 4]
                    selected.append(partial)
                break
        return selected

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------

    def _score_sentences(self, sentences: list[str], query: str) -> list[float]:
        if self.strategy == "perplexity":
            return self._perplexity_scores(sentences)
        elif self.strategy == "extractive":
            return self._tfidf_scores(sentences, query)
        else:
            # truncate: keep first sentences (descending index = higher priority)
            return list(range(len(sentences), 0, -1))

    def _perplexity_scores(self, sentences: list[str]) -> list[float]:
        """Higher perplexity = more surprising = keep."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._model is None:
                logger.info("Loading proxy LM: %s", self.proxy_model)
                self._tokenizer = AutoTokenizer.from_pretrained(self.proxy_model)
                self._model = AutoModelForCausalLM.from_pretrained(self.proxy_model)
                self._model.eval()

            scores = []
            for sent in sentences:
                enc = self._tokenizer(sent, return_tensors="pt")
                with torch.no_grad():
                    loss = self._model(**enc, labels=enc["input_ids"]).loss
                scores.append(float(loss))
            return scores

        except Exception as e:
            logger.warning("Perplexity scoring failed (%s), falling back to TF-IDF", e)
            return self._tfidf_scores(sentences, "")

    def _tfidf_scores(self, sentences: list[str], query: str) -> list[float]:
        """TF-IDF cosine similarity to query + sentence length heuristic."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            corpus = sentences + ([query] if query else [])
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf = vec.fit_transform(corpus)

            if query:
                query_vec = tfidf[-1]
                sent_vecs = tfidf[:-1]
                scores = cosine_similarity(sent_vecs, query_vec).ravel().tolist()
            else:
                # fallback: score by sentence length (longer = more info)
                scores = [len(s.split()) for s in sentences]
            return scores
        except ImportError:
            return [len(s.split()) for s in sentences]

    def _select_sentences(self, sentences: list[str], scores: list[float]) -> str:
        n_keep = max(self.min_sentences, int(len(sentences) * self.target_ratio))
        ranked = sorted(
            enumerate(sentences),
            key=lambda x: scores[x[0]],
            reverse=True,
        )
        kept_indices = sorted(idx for idx, _ in ranked[:n_keep])
        return " ".join(sentences[i] for i in kept_indices)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _count_tokens(text: str) -> int:
        # rough approximation: 4 chars per token
        return max(1, len(text) // 4)
