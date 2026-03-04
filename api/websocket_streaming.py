"""
WebSocket streaming inference - real-time token-by-token output.
Covers: WebSockets, real-time applications
"""
import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Dict, Optional

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


async def stream_rag_response(query: str) -> AsyncIterator[str]:
    """
    Full RAG pipeline with streaming LLM output.
    Retrieval + reranking are delegated to the shared orchestrator singleton
    so only one copy of each model is loaded per process.
    """
    from agents.orchestrator import get_pipeline

    pipeline = get_pipeline()
    chunks = await asyncio.to_thread(pipeline.retriever.retrieve, query)
    reranked = await asyncio.to_thread(pipeline.reranker.rerank, query, chunks)

    context_str = ""
    for i, chunk in enumerate(reranked):
        context_str += f"[source_{i+1}] {chunk['source']}:\n{chunk['text']}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
    ]

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
                async for token in stream_rag_response(query):
                    full_response += token
                    token_count += 1
                    await manager.send(
                        conn_id,
                        {
                            "type": "token",
                            "content": token,
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

        while True:
            status = get_job_status(job_id)
            await manager.send(conn_id, {"type": "progress", **status})
            if status.get("status") in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        await manager.disconnect(conn_id)
