"""
FastAPI router for the multi-agent system.

Endpoints:
  POST /multi-agent/query   - Submit a query to the multi-agent pipeline
  GET  /multi-agent/status  - Health and circuit breaker status
  GET  /multi-agent/hitl    - List pending human-in-the-loop requests
  POST /multi-agent/hitl/{session_id}/resolve - Resolve a HITL request

Mount in api/main.py:
    from agents.multi_agent.api_router import router as multi_agent_router
    app.include_router(multi_agent_router, prefix="/multi-agent", tags=["multi-agent"])
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from agents.multi_agent.supervisor import Supervisor

logger = logging.getLogger(__name__)
router = APIRouter()

_supervisor: Optional[Supervisor] = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor(quality_threshold=0.75, max_iterations=3)
    return _supervisor


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The user query.")
    session_id: Optional[str] = Field(None, description="Optional session ID for memory continuity.")
    priority: str = Field("normal", description="Task priority: low, normal, high, critical.")
    max_iterations: Optional[int] = Field(None, ge=1, le=5)


class AgentResultResponse(BaseModel):
    agent_name: str
    status: str
    confidence: float
    output_preview: str
    latency_ms: float
    error: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    query: str
    final_answer: str
    confidence: float
    agreement_score: float
    consensus_strategy: str
    iterations: int
    total_latency_ms: float
    agent_results: List[AgentResultResponse]
    hitl_triggered: bool
    hitl_reason: Optional[str] = None


class HITLResolveRequest(BaseModel):
    resolution: str = Field(..., min_length=1, max_length=2048)


class StatusResponse(BaseModel):
    status: str
    agents: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    hitl_queue_size: int
    pending_hitl: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Submit a query to the multi-agent pipeline")
async def multi_agent_query(request: QueryRequest) -> QueryResponse:
    """
    Routes the query through the Research -> Critic -> Verifier pipeline.

    The supervisor:
    1. Classifies query complexity and selects agents
    2. Runs research + critique loop until quality threshold is met
    3. Verifies factual claims against retrieved sources
    4. Returns consensus answer with confidence score

    If confidence < 0.55 or safety keywords detected, HITL is triggered.
    """
    supervisor = get_supervisor()

    if request.max_iterations:
        supervisor.max_iterations = request.max_iterations

    try:
        trace = supervisor.run(request.query, session_id=request.session_id)
    except Exception as e:
        logger.error("Multi-agent pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    agent_responses = [
        AgentResultResponse(
            agent_name=r.agent_name,
            status=r.status.value,
            confidence=round(r.confidence, 3),
            output_preview=r.output[:200] if r.output else "",
            latency_ms=round(r.latency_ms, 1),
            error=r.error,
        )
        for r in trace.agent_results
    ]

    return QueryResponse(
        session_id=trace.session_id,
        query=trace.query,
        final_answer=trace.final_answer,
        confidence=round(trace.consensus.confidence if trace.consensus else 0.0, 3),
        agreement_score=round(trace.consensus.agreement_score if trace.consensus else 0.0, 3),
        consensus_strategy=trace.consensus.strategy if trace.consensus else "none",
        iterations=trace.iterations,
        total_latency_ms=round(trace.total_latency_ms, 1),
        agent_results=agent_responses,
        hitl_triggered=trace.hitl_triggered,
        hitl_reason=trace.hitl_request.reason if trace.hitl_request else None,
    )


@router.get("/status", response_model=StatusResponse, summary="Get multi-agent system health")
async def multi_agent_status() -> StatusResponse:
    """Returns health metrics for all agents and circuit breaker states."""
    supervisor = get_supervisor()
    health = supervisor.health()
    return StatusResponse(
        status="ok",
        agents=health["agents"],
        circuit_breakers=health["circuit_breakers"],
        hitl_queue_size=health["hitl_queue_size"],
        pending_hitl=health["pending_hitl"],
    )


@router.get("/hitl", summary="List pending human-in-the-loop requests")
async def list_hitl_requests() -> List[Dict[str, Any]]:
    """Returns all unresolved HITL requests requiring human review."""
    supervisor = get_supervisor()
    return [
        {
            "session_id": h.session_id,
            "task_id": h.task_id,
            "reason": h.reason,
            "confidence": round(h.confidence, 3),
            "output_preview": h.agent_output[:200],
            "timestamp": h.timestamp,
            "resolved": h.resolved,
        }
        for h in supervisor._hitl_queue
        if not h.resolved
    ]


@router.post("/hitl/{session_id}/resolve", summary="Resolve a HITL request")
async def resolve_hitl(session_id: str, request: HITLResolveRequest) -> Dict[str, Any]:
    """
    Marks a HITL request as resolved with a human-provided response.
    The resolution is stored and can be used to improve future responses.
    """
    supervisor = get_supervisor()
    resolved = supervisor.resolve_hitl(session_id, request.resolution)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"No pending HITL request for session '{session_id}'.",
        )
    return {"status": "resolved", "session_id": session_id}
