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


def _constitutional_api_gate(text: str, endpoint: str) -> None:
    """Mitchell-style + behavioral pre-check before LLM-backed paths (opt-in)."""
    if os.getenv("ENABLE_API_CONSTITUTIONAL_GATE", "").lower() not in ("1", "true", "yes"):
        return
    from safety.constitutional_filter import (
        audit_gateway_decision,
        constitutional_filter_query,
    )

    r = constitutional_filter_query(text)
    audit_gateway_decision(
        endpoint=endpoint,
        allowed=r.allowed,
        reason=r.reason,
        extra={"text_chars": len(text)},
    )
    if not r.allowed:
        raise HTTPException(status_code=400, detail=r.to_http_detail())


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
    session_id: str | None = Field(
        default=None,
        max_length=256,
        description="Optional long-running research session for ResearchLog injection.",
    )


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
    _constitutional_api_gate(request.query, "/retrieve")
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
    _constitutional_api_gate(request.query, "/query")
    result = await asyncio.to_thread(run_pipeline, request.query, request.session_id)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    resp = result["response"]
    out: Dict[str, Any] = {
        "answer": resp["answer"],
        "sources": resp.get("sources") or [],
        "tokens_used": resp.get("tokens_used", 0),
    }
    if resp.get("behavioral_blocked") is not None:
        out["behavioral_blocked"] = resp["behavioral_blocked"]
    if resp.get("behavioral_reasons") is not None:
        out["behavioral_reasons"] = resp["behavioral_reasons"]
    if resp.get("grounding_confidence") is not None:
        out["grounding_confidence"] = resp["grounding_confidence"]
    if resp.get("source_attributions") is not None:
        out["source_attributions"] = resp["source_attributions"]
        out["speculative_sentence_count"] = resp.get("speculative_sentence_count", 0)
    if resp.get("constitutional_score") is not None:
        out["constitutional_score"] = resp["constitutional_score"]
        out["constitutional_passed"] = resp.get("constitutional_passed")
    if resp.get("quality_alert"):
        out["quality_alert"] = resp["quality_alert"]
    if resp.get("effective_faithfulness") is not None:
        out["effective_faithfulness"] = resp["effective_faithfulness"]
    if resp.get("answer_with_provenance"):
        out["answer_with_provenance"] = resp["answer_with_provenance"]
    if resp.get("adversarial_consensus") is not None:
        out["adversarial_consensus"] = resp["adversarial_consensus"]
    if resp.get("consensus_hitl") is not None:
        out["consensus_hitl"] = resp["consensus_hitl"]
    if resp.get("consensus_score") is not None:
        out["consensus_score"] = resp["consensus_score"]
    if resp.get("consensus_discrepancy") is not None:
        out["consensus_discrepancy"] = resp["consensus_discrepancy"]
    if resp.get("truth_committee") is not None:
        out["truth_committee"] = resp["truth_committee"]
    return out


@app.post("/batch", dependencies=[Security(_require_api_key)])
async def query_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    for q in request.queries:
        _constitutional_api_gate(q, "/batch")
    job_id = str(uuid.uuid4())
    # BackgroundTasks runs sync callables in a thread pool automatically,
    # so enqueue_batch_job (which does blocking Redis + Celery I/O) is safe here.
    background_tasks.add_task(enqueue_batch_job, job_id, request.queries)
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
