"""
Lightweight realtime API -- delegates to the shared Pipeline.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agents.orchestrator import run_pipeline


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI(title="LLMOps Research Assistant Realtime API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    result = run_pipeline(req.query)
    if result.get("error"):
        return {"error": result["error"]}
    return {
        "answer": result["response"]["answer"],
        "sources": result["response"]["sources"],
        "tokens_used": result["response"]["tokens_used"],
    }
