"""
FastAPI gateway - realtime + batch inference endpoints.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import run_pipeline
from api.websocket_streaming import router as websocket_router

logger = logging.getLogger(__name__)

app = FastAPI(title="LLMOps Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)


class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class BatchRequest(BaseModel):
    queries: list[str]


@app.post("/query")
async def query_realtime(request: QueryRequest):
    result = await asyncio.to_thread(run_pipeline, request.query)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    return {
        "answer": result["response"]["answer"],
        "sources": result["response"]["sources"],
        "tokens_used": result["response"]["tokens_used"],
    }


@app.post("/batch")
async def query_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    from api.batch import enqueue_batch_job

    job_id = str(uuid.uuid4())
    background_tasks.add_task(enqueue_batch_job, job_id, request.queries)
    return {"job_id": job_id, "status": "queued"}


@app.get("/batch/{job_id}")
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
