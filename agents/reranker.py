"""
Two-stage retrieval: bi-encoder dense embeddings + cross-encoder reranking.

Stage 1: BiEncoderEmbedder (fast ~5ms)
  - Mean-pool over last_hidden_state with attention masking
  - L2-normalize for cosine similarity via inner product

Stage 2: CrossEncoderReranker (precise ~40ms)
  - Encodes (query, doc) pairs jointly through a cross-encoder
  - Logit-based relevance scoring -> global re-sort -> top-k
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

BIENCODER_MODEL = os.getenv(
    "BIENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CROSSENCODER_MODEL = os.getenv(
    "CROSSENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class BiEncoderEmbedder:
    """Mean-pool embedding model for dense retrieval (Stage 1)."""

    def __init__(self, model_name: str = BIENCODER_MODEL, device: str | None = None):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._F = F
        self._np = np
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("BiEncoder loaded: %s on %s", model_name, self.device)

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        import torch

        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    def embed(self, texts: List[str], batch_size: int = 64):
        import numpy as np

        all_embeddings = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                output = self.model(**encoded)
                embeddings = self._mean_pool(output, encoded["attention_mask"])
                embeddings = self._F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)


class CrossEncoderReranker:
    """Cross-encoder relevance scorer for Stage 2 reranking."""

    def __init__(self, model_name: str = CROSSENCODER_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("CrossEncoder loaded: %s on %s", model_name, self.device)

    def score_pairs(
        self, query: str, documents: List[str], batch_size: int = 32
    ) -> List[float]:
        all_scores: List[float] = []
        with self._torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                pairs = [(query, doc) for doc in batch_docs]
                encoded = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**encoded).logits.squeeze(-1)
                # Normalize raw logits to [0, 1] via sigmoid so rerank_score
                # is comparable to retrieval_score (cosine similarity in [0, 1]).
                scores = self._torch.sigmoid(logits).cpu().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                all_scores.extend(scores)

        return all_scores


class RerankerAgent:
    """
    Reranker using cross-encoder relevance scoring.
    Falls back to passthrough (truncation) when the model is unavailable.
    """

    def __init__(
        self,
        top_k: int = 5,
        crossencoder_model: str = CROSSENCODER_MODEL,
    ):
        self.top_k = top_k
        self._ready = False
        self.cross_encoder: CrossEncoderReranker | None = None

        try:
            self.cross_encoder = CrossEncoderReranker(model_name=crossencoder_model)
            self._ready = True
        except Exception as exc:
            logger.warning("CrossEncoder unavailable, using passthrough: %s", exc)

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self._ready or not candidates or self.cross_encoder is None:
            return candidates[: self.top_k]

        documents = [c["text"] for c in candidates]
        scores = self.cross_encoder.score_pairs(query, documents)

        # Shallow-copy each dict so the caller's list is not mutated in-place.
        scored = [{**c, "rerank_score": scores[i]} for i, c in enumerate(candidates)]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: self.top_k]
