"""
Retriever Agent - semantic search over FAISS index.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ingestion.pipeline import EMBED_MODEL, INDEX_PATH, META_PATH, EmbeddingModel

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 0.5


@dataclass
class RetrievedChunk:
    text: str
    source: str
    retrieval_score: float
    rerank_score: Optional[float] = None


class RetrieverAgent:
    def __init__(self, top_k: int = 10, shards_path: str = "data/shards"):
        self.embedder = EmbeddingModel()
        self.top_k = top_k
        self.aggregator_url = os.getenv("SHARD_AGGREGATOR_URL", "").strip()
        self.use_microservice_distributed = bool(self.aggregator_url)
        self.use_distributed = False
        self.distributed_index = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        if self.use_microservice_distributed:
            return

        try:
            from infra.distributed_index import DistributedFAISSIndex

            self.distributed_index = DistributedFAISSIndex()
            if os.path.exists(shards_path):
                self.distributed_index.load(shards_path)
                self.use_distributed = len(self.distributed_index.shards) > 0
        except Exception:
            self.distributed_index = None
            self.use_distributed = False

        if not self.use_distributed:
            if not os.path.exists(INDEX_PATH):
                raise FileNotFoundError(
                    f"FAISS index not found at {INDEX_PATH}. Run the ingestion pipeline first."
                )
            import faiss

            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Loaded local FAISS index with %d vectors", self.index.ntotal)

    def _aggregator_search(self, query_vec: list[float]) -> List[Dict[str, Any]]:
        payload = {"query_vector": query_vec, "top_k": self.top_k}
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    f"{self.aggregator_url}/search", json=payload, timeout=15.0
                )
                resp.raise_for_status()
                return resp.json().get("results", [])
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                logger.warning(
                    "Aggregator attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
                )
                time.sleep(RETRY_BACKOFF_S * attempt)
        raise RuntimeError(f"Aggregator unreachable after {MAX_RETRIES} retries") from last_exc

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed([query])

        if self.use_microservice_distributed:
            return self._aggregator_search(query_vec[0].tolist())

        if self.use_distributed and self.distributed_index is not None:
            return self.distributed_index.search(query_vec, top_k=self.top_k)

        assert self.index is not None, "No FAISS index loaded"
        scores, indices = self.index.search(query_vec, self.top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx].copy()
            chunk["retrieval_score"] = float(score)
            results.append(chunk)

        return results
