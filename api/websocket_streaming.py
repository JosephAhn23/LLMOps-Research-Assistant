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
client = AsyncOpenAI()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        conn_id = str(uuid.uuid4())
        self.active_connections[conn_id] = websocket
        return conn_id

    def disconnect(self, conn_id: str):
        self.active_connections.pop(conn_id, None)

    async def send(self, conn_id: str, data: dict):
        ws = self.active_connections.get(conn_id)
        if ws:
            await ws.send_text(json.dumps(data))


manager = ConnectionManager()
_retriever = None
_reranker = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from agents.retriever import RetrieverAgent

        _retriever = RetrieverAgent(top_k=10)
    return _retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from agents.reranker import RerankerAgent

        _reranker = RerankerAgent(top_k=5)
    return _reranker


async def stream_rag_response(query: str) -> AsyncIterator[str]:
    """
    Full RAG pipeline with streaming LLM output.
    Retrieval + reranking happen upfront, then LLM streams tokens.
    """
    retriever = _get_retriever()
    reranker = _get_reranker()
    chunks = retriever.retrieve(query)
    reranked = reranker.rerank(query, chunks)

    context_str = ""
    for i, chunk in enumerate(reranked):
        context_str += f"[source_{i+1}] {chunk['source']}:\n{chunk['text']}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
    ]

    stream = await client.chat.completions.create(
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
        manager.disconnect(conn_id)


@router.websocket("/ws/batch-progress")
async def websocket_batch_progress(websocket: WebSocket):
    """
    Real-time batch job progress streaming.
    """
    conn_id = await manager.connect(websocket)

    try:
        raw = await websocket.receive_text()
        request = json.loads(raw)
        job_id = request.get("job_id")

        if not job_id:
            await manager.send(conn_id, {"type": "error", "message": "missing job_id"})
            return

        from api.batch import get_job_status

        while True:
            status = get_job_status(job_id)
            await manager.send(conn_id, {"type": "progress", **status})
            if status.get("status") in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        manager.disconnect(conn_id)
