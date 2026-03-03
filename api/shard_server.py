"""
Shard-local retrieval service for distributed FAISS microservice topology.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FAISS Shard Server")

SHARD_ID = int(os.getenv("SHARD_ID", "0"))
SHARDS_PATH = Path(os.getenv("SHARDS_PATH", "data/shards"))
INDEX_PATH = SHARDS_PATH / f"shard_{SHARD_ID}.index"
META_PATH = SHARDS_PATH / f"shard_{SHARD_ID}.meta.json"

index = None
metadata = []


class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10


@app.on_event("startup")
def startup() -> None:
    global index, metadata
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Shard index not found: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "shard_id": SHARD_ID}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    if index is None:
        return {"results": []}
    query = np.asarray([req.query_vector], dtype=np.float32)
    scores, indices = index.search(query, req.top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        chunk = metadata[idx].copy()
        chunk["retrieval_score"] = float(score)
        chunk["shard"] = SHARD_ID
        results.append(chunk)
    return {"results": results}
