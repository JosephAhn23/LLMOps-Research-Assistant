"""
Distributed FAISS microservice architecture.
Each shard is a separate FastAPI process. Aggregator fans out with asyncio.gather.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List

import faiss
import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from ingestion.pipeline import EmbeddingModel

logger = logging.getLogger(__name__)

# ─── Shard Server (runs as separate process per shard) ────────

shard_app = FastAPI()
embedder = EmbeddingModel()
shard_index = None
shard_metadata = []
SHARD_ID = int(os.environ.get("SHARD_ID", 0))
SHARD_DATA_PATH = f"data/shards/shard_{SHARD_ID}"


@shard_app.on_event("startup")
def load_shard():
    global shard_index, shard_metadata
    shard_index = faiss.read_index(f"{SHARD_DATA_PATH}.index")
    with open(f"{SHARD_DATA_PATH}_meta.json", encoding="utf-8") as f:
        shard_metadata = json.load(f)
    logger.info("Shard %d loaded: %d vectors", SHARD_ID, shard_index.ntotal)


class ShardSearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10


class ShardSearchResult(BaseModel):
    chunks: List[Dict]
    shard_id: int


@shard_app.post("/search", response_model=ShardSearchResult)
def search_shard(request: ShardSearchRequest):
    query = np.array([request.query_vector], dtype=np.float32)
    scores, indices = shard_index.search(query, request.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(shard_metadata):
            continue
        chunk = shard_metadata[idx].copy()
        chunk["retrieval_score"] = float(score)
        chunk["shard_id"] = SHARD_ID
        results.append(chunk)

    return ShardSearchResult(chunks=results, shard_id=SHARD_ID)


@shard_app.get("/health")
def health():
    return {"shard_id": SHARD_ID, "vectors": shard_index.ntotal if shard_index else 0}


# ─── Aggregator Server (single entry point) ───────────────────

aggregator_app = FastAPI()

_DEFAULT_SHARD_URLS = "http://shard-0:8001,http://shard-1:8002,http://shard-2:8003,http://shard-3:8004"
SHARD_URLS = [u.strip() for u in os.getenv("SHARD_URLS", _DEFAULT_SHARD_URLS).split(",") if u.strip()]


class AggregatorSearchRequest(BaseModel):
    query: str
    top_k: int = 10


@aggregator_app.post("/search")
async def distributed_search(request: AggregatorSearchRequest):
    """
    Fan out query to all shards in parallel via asyncio.gather.
    Merge and re-rank results globally.
    """
    query_vec = embedder.embed([request.query])[0].tolist()

    shard_request = {"query_vector": query_vec, "top_k": request.top_k}

    async def search_one_shard(url: str, client: httpx.AsyncClient) -> List[Dict]:
        try:
            response = await client.post(
                f"{url}/search",
                json=shard_request,
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()["chunks"]
        except Exception as e:
            logger.warning("Shard %s failed: %s", url, e)
            return []

    async with httpx.AsyncClient() as client:
        shard_results = await asyncio.gather(
            *[search_one_shard(url, client) for url in SHARD_URLS],
            return_exceptions=False,
        )

    all_chunks = [chunk for shard in shard_results for chunk in shard]
    all_chunks.sort(key=lambda x: x["retrieval_score"], reverse=True)
    top_chunks = all_chunks[: request.top_k]

    return {
        "chunks": top_chunks,
        "shards_queried": len(SHARD_URLS),
        "total_candidates": len(all_chunks),
    }


@aggregator_app.get("/health")
async def aggregator_health():
    """Check all shards are alive."""

    async with httpx.AsyncClient() as client:
        async def check(url):
            try:
                r = await client.get(f"{url}/health", timeout=2.0)
                return {"url": url, "status": "ok", **r.json()}
            except Exception:
                return {"url": url, "status": "down"}

        shard_statuses = await asyncio.gather(*[check(url) for url in SHARD_URLS])

    return {
        "aggregator": "ok",
        "shards": shard_statuses,
        "healthy_shards": sum(1 for s in shard_statuses if s["status"] == "ok"),
    }


# ─── Index Builder (run once to partition + save shards) ──────

def build_shard_indexes(embeddings: np.ndarray, metadata: List[Dict], n_shards: int = 4):
    """Partition corpus into shards and save each independently."""
    os.makedirs("data/shards", exist_ok=True)

    n = len(embeddings)
    shard_size = n // n_shards

    for i in range(n_shards):
        start = i * shard_size
        end = n if i == n_shards - 1 else (i + 1) * shard_size

        shard_emb = embeddings[start:end].astype(np.float32)
        shard_meta = metadata[start:end]

        # IVFFlat for approximate search on each shard
        dim = shard_emb.shape[1]
        nlist = min(100, len(shard_emb) // 10)
        quantizer = faiss.IndexFlatIP(dim)

        if len(shard_emb) >= nlist * 10:
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(shard_emb)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(shard_emb)
        faiss.write_index(index, f"data/shards/shard_{i}.index")

        with open(f"data/shards/shard_{i}_meta.json", "w", encoding="utf-8") as f:
            json.dump(shard_meta, f)

        logger.info("Shard %d: %d vectors saved", i, len(shard_emb))
