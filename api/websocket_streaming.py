"""
WebSocket streaming inference - real-time token-by-token output.
Covers: WebSockets, real-time applications
"""
import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from agents.synthesizer import SYSTEM_PROMPT

router = APIRouter()

# Lazy-initialized so that missing OPENAI_API_KEY at import time does not
# crash the entire FastAPI app during test collection or Docker build.
_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        conn_id = str(uuid.uuid4())
        async with self._lock:
            self.active_connections[conn_id] = websocket
        return conn_id

    async def disconnect(self, conn_id: str):
        async with self._lock:
            self.active_connections.pop(conn_id, None)

    async def send(self, conn_id: str, data: dict):
        ws = self.active_connections.get(conn_id)
        if ws:
            await ws.send_text(json.dumps(data))


manager = ConnectionManager()


async def prepare_rag_stream_context(query: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Retrieve + rerank + Chain-of-Thought scaffold for glass-box WebSocket clients.

    Returns:
        thinking: scratchpad-style CoT prompt text (shown before tokens).
        reranked: chunk dicts for source sidecar.
        messages: OpenAI chat messages for streaming completion.
    """
    from agents.orchestrator import get_pipeline
    from context_engineering.chain_of_thought import ChainOfThoughtBuilder

    pipeline = get_pipeline()
    chunks = await asyncio.to_thread(pipeline.retriever.retrieve, query)
    reranked = await asyncio.to_thread(pipeline.reranker.rerank, query, chunks)

    context_str = ""
    for i, chunk in enumerate(reranked):
        context_str += f"[source_{i+1}] {chunk['source']}:\n{chunk['text']}\n\n"

    try:
        cot = ChainOfThoughtBuilder()
        thinking = cot.build(query, context_str, strategy="scratchpad")
    except Exception:
        thinking = f"[CoT] Query: {query[:500]}\n\nContext length: {len(context_str)} chars."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
    ]
    return thinking, reranked, messages


async def stream_rag_response(query: str) -> AsyncIterator[str]:
    """
    Full RAG pipeline with streaming LLM output.
    Retrieval + reranking are delegated to the shared orchestrator singleton
    so only one copy of each model is loaded per process.
    """
    _thinking, _reranked, messages = await prepare_rag_stream_context(query)

    stream = await _get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming inference.
    """
    conn_id = await manager.connect(websocket)

    try:
        while True:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send(conn_id, {"type": "error", "message": "invalid JSON"})
                continue
            query = request.get("query", "")

            if not query:
                await manager.send(conn_id, {"type": "error", "message": "empty query"})
                continue

            start = time.perf_counter()
            full_response = ""
            token_count = 0
            await manager.send(conn_id, {"type": "status", "message": "retrieving context..."})

            try:
                thinking, reranked, messages = await prepare_rag_stream_context(query)
                await manager.send(
                    conn_id,
                    {"type": "thinking", "content": thinking[:16000]},
                )
                source_payload = []
                for i, ch in enumerate(reranked):
                    source_payload.append(
                        {
                            "index": i + 1,
                            "source": ch.get("source", f"doc_{i + 1}"),
                            "text_preview": (ch.get("text") or "")[:1500],
                            "retrieval_score": ch.get("retrieval_score"),
                            "rerank_score": ch.get("rerank_score"),
                            "doc_id": ch.get("doc_id"),
                            "ingested_at": ch.get("ingested_at"),
                        }
                    )
                await manager.send(conn_id, {"type": "sources", "chunks": source_payload})
                await manager.send(
                    conn_id,
                    {
                        "type": "consensus",
                        "status": "stream_mode",
                        "detail": (
                            "Token stream uses single-model synthesis. "
                            "Use POST /query or the Gradio Transparent (REST) mode for "
                            "truth-committee / adversarial consensus metadata."
                        ),
                    },
                )
                await manager.send(
                    conn_id,
                    {
                        "type": "sandbox",
                        "log": (
                            "[sandbox] No code execution on this path. "
                            "Wire SelfHealingLoop / StrictExecutionOrchestrator to stream logs here."
                        ),
                    },
                )

                stream = await _get_openai_client().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if not delta.content:
                        continue
                    token_count += 1
                    full_response += delta.content
                    await manager.send(
                        conn_id,
                        {
                            "type": "token",
                            "content": delta.content,
                            "token_index": token_count,
                        },
                    )

                latency_ms = round((time.perf_counter() - start) * 1000, 1)
                await manager.send(
                    conn_id,
                    {
                        "type": "done",
                        "full_response": full_response,
                        "token_count": token_count,
                        "latency_ms": latency_ms,
                        "tokens_per_second": round(token_count / max(latency_ms / 1000, 1e-6), 1),
                    },
                )
            except Exception as e:
                await manager.send(conn_id, {"type": "error", "message": str(e)})

    except (WebSocketDisconnect, asyncio.TimeoutError):
        await manager.disconnect(conn_id)


@router.websocket("/ws/batch-progress")
async def websocket_batch_progress(websocket: WebSocket):
    """
    Real-time batch job progress streaming.
    """
    conn_id = await manager.connect(websocket)

    try:
        raw = await websocket.receive_text()
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            await manager.send(conn_id, {"type": "error", "message": "invalid JSON"})
            await manager.disconnect(conn_id)
            return
        job_id = request.get("job_id")

        if not job_id:
            await manager.send(conn_id, {"type": "error", "message": "missing job_id"})
            await manager.disconnect(conn_id)
            return

        from api.batch import get_job_status

        _MAX_POLL_S = float(os.getenv("BATCH_PROGRESS_TIMEOUT_S", "300"))
        deadline = asyncio.get_event_loop().time() + _MAX_POLL_S

        while True:
            status = get_job_status(job_id)
            if status.get("error"):
                await manager.send(conn_id, {"type": "error", "message": status["error"]})
                break
            await manager.send(conn_id, {"type": "progress", **status})
            if status.get("status") in ["completed", "failed"]:
                break
            if asyncio.get_event_loop().time() >= deadline:
                await manager.send(
                    conn_id, {"type": "error", "message": "polling timeout — job may be stuck"}
                )
                break
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        await manager.disconnect(conn_id)
