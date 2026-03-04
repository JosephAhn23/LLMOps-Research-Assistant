"""
FastAPI gateway - realtime + batch inference endpoints.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict

from agents.orchestrator import run_pipeline
from api.batch import enqueue_batch_job
from api.websocket_streaming import router as websocket_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — restrict to an explicit allowlist in production.
# Set CORS_ORIGINS="https://app.example.com,https://admin.example.com" to
# override. Falls back to localhost-only for local development.
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
_allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Optional API-key authentication.
# Set API_KEY env var to require a bearer key on all mutating endpoints.
# When unset, auth is skipped (dev mode).
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(title="LLMOps Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(websocket_router)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    stream: bool = False


class BatchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=100)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = True


class IngestRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 512

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v


@app.post("/retrieve", dependencies=[Security(_require_api_key)])
async def retrieve(request: RetrieveRequest):
    """Retrieve and optionally rerank documents — used by the MCP server."""
    from agents.orchestrator import get_pipeline

    pipeline = get_pipeline()
    chunks = await asyncio.to_thread(pipeline.retriever.retrieve, request.query)
    if request.rerank:
        chunks = await asyncio.to_thread(pipeline.reranker.rerank, request.query, chunks)
    return {"results": chunks[: request.top_k]}


_ingestion_pipeline = None
_ingestion_lock = threading.Lock()


def _get_ingestion_pipeline():
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        with _ingestion_lock:
            if _ingestion_pipeline is None:
                from ingestion.pipeline import IngestionPipeline
                _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


@app.post("/ingest", dependencies=[Security(_require_api_key)])
async def ingest(request: IngestRequest):
    """Ingest a document into the FAISS index — used by the MCP server."""
    doc_id = str(uuid.uuid4())
    source = (request.metadata or {}).get("source", doc_id)
    pipeline = _get_ingestion_pipeline()
    await asyncio.to_thread(
        pipeline.ingest_documents,
        [{"id": doc_id, "text": request.content, "source": source}],
    )
    return {"status": "ingested", "doc_id": doc_id, "source": source}


@app.post("/query", dependencies=[Security(_require_api_key)])
async def query_realtime(request: QueryRequest):
    result = await asyncio.to_thread(run_pipeline, request.query)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    return {
        "answer": result["response"]["answer"],
        "sources": result["response"]["sources"],
        "tokens_used": result["response"]["tokens_used"],
    }


@app.post("/batch", dependencies=[Security(_require_api_key)])
async def query_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # enqueue_batch_job performs synchronous Redis + Celery I/O; run it in a
    # thread so it does not block the async event loop.
    background_tasks.add_task(asyncio.to_thread, enqueue_batch_job, job_id, request.queries)
    return {"job_id": job_id, "status": "queued"}


@app.get("/batch/{job_id}", dependencies=[Security(_require_api_key)])
async def get_batch_status(job_id: str):
    from api.batch import get_job_status

    status = get_job_status(job_id)
    if status.get("error"):
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@app.get("/health")
async def health():
    checks: dict[str, str] = {"api": "ok"}

    try:
        from api.batch import redis_client

        redis_client.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
