"""
Microservices Architecture — decomposes the monolith into independent services.

Components:
  - ServiceRegistry   — class-level URL registry (swap for Consul/K8s DNS in prod)
  - EventType         — enum of all pipeline events
  - Event             — dataclass with serialisation helpers
  - EventBus          — async pub/sub: aiokafka (prod) / aioredis Streams (local) / memory (test)
  - BaseMicroservice  — FastAPI base with /health and /metrics
  - RetrievalService  — wraps agents/retriever.py (port 8001)
  - RerankerService   — wraps agents/reranker.py (port 8002)
  - SafetyService     — wraps safety/semantic_safety.py (port 8004)
  - APIGateway        — HTTP orchestration: safety → retrieve → rerank → synthesize (port 8000)

Existing docker-compose.yml stays for local monolith mode.
docker-compose.microservices.yml is the distributed version.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    QUERY_RECEIVED = "query.received"
    RETRIEVAL_COMPLETE = "retrieval.complete"
    RERANK_COMPLETE = "rerank.complete"
    GENERATION_COMPLETE = "generation.complete"
    SAFETY_FLAGGED = "safety.flagged"
    SAFETY_CHECK_PASSED = "safety.check.passed"
    SAFETY_CHECK_FAILED = "safety.check.failed"
    EVAL_COMPLETE = "eval.complete"
    INGEST_COMPLETE = "ingest.complete"
    ERROR = "pipeline.error"


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_service: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "type": self.type.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "source_service": self.source_service,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> "Event":
        d = json.loads(data)
        return cls(
            type=EventType(d["type"]),
            payload=d["payload"],
            correlation_id=d.get("correlation_id", ""),
            source_service=d.get("source_service", ""),
            event_id=d.get("event_id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """
    Class-level URL registry — no instantiation required in the gateway.
    In production, swap for Consul service discovery or Kubernetes DNS.
    """

    _registry: Dict[str, str] = {
        "retrieval":  os.getenv("RETRIEVAL_SVC_URL",  "http://retrieval-svc:8001"),
        "reranker":   os.getenv("RERANKER_SVC_URL",   "http://reranker-svc:8002"),
        "synthesizer":os.getenv("SYNTHESIZER_SVC_URL","http://synthesizer-svc:8003"),
        "safety":     os.getenv("SAFETY_SVC_URL",     "http://safety-svc:8004"),
        "evaluation": os.getenv("EVALUATION_SVC_URL", "http://evaluation-svc:8005"),
        "multimodal": os.getenv("MULTIMODAL_SVC_URL", "http://multimodal-svc:8006"),
    }

    @classmethod
    def get(cls, service: str) -> str:
        if service not in cls._registry:
            raise KeyError(f"Unknown service: '{service}'")
        return cls._registry[service]

    @classmethod
    def register(cls, service: str, url: str) -> None:
        cls._registry[service] = url
        logger.info("Registered service: %s → %s", service, url)

    @classmethod
    def list_services(cls) -> Dict[str, str]:
        return dict(cls._registry)

    @classmethod
    async def health_check_all(cls) -> Dict[str, bool]:
        results = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in cls._registry.items():
                try:
                    resp = await client.get(f"{url}/health")
                    results[name] = resp.status_code == 200
                except Exception:
                    results[name] = False
        return results


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Async pub/sub event bus.

    Backend auto-selection:
      KAFKA_BROKERS env set  → aiokafka (production)
      REDIS_URL env set      → aioredis Streams (local/dev)
      Neither                → in-memory handlers only (testing)

    subscribe() can be used as a decorator or called directly.
    consume() is an async generator for Redis Streams consumption.
    """

    def __init__(self, backend: Optional[str] = None):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self.backend = backend or self._detect_backend()

    def _detect_backend(self) -> str:
        if os.environ.get("KAFKA_BROKERS") or os.environ.get("KAFKA_BOOTSTRAP_SERVERS"):
            return "kafka"
        if os.environ.get("REDIS_URL"):
            return "redis"
        return "memory"

    def subscribe(self, event_type: EventType) -> Callable:
        """Decorator to register an async or sync event handler."""
        def decorator(func: Callable) -> Callable:
            self._handlers.setdefault(event_type, []).append(func)
            logger.debug("Subscribed %s to %s", func.__name__, event_type.value)
            return func
        return decorator

    def subscribe_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register a handler directly (non-decorator form)."""
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        """Publish to in-process handlers and the configured external backend."""
        for handler in self._handlers.get(event.type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error("Event handler error (%s): %s", handler.__name__, e)

        if self.backend == "kafka":
            await self._publish_kafka(event)
        elif self.backend == "redis":
            await self._publish_redis(event)

    async def _publish_kafka(self, event: Event) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            brokers = os.environ.get("KAFKA_BROKERS") or os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
            )
            producer = AIOKafkaProducer(bootstrap_servers=brokers)
            await producer.start()
            try:
                await producer.send_and_wait(
                    event.type.value,
                    event.to_json().encode(),
                )
            finally:
                await producer.stop()
        except Exception as e:
            logger.error("Kafka publish failed: %s", e)

    async def _publish_redis(self, event: Event) -> None:
        try:
            import aioredis
            redis = await aioredis.from_url(
                os.environ.get("REDIS_URL", "redis://localhost:6379")
            )
            await redis.xadd(
                f"llmops:events:{event.type.value}",
                {"data": event.to_json()},
                maxlen=10000,
            )
            await redis.aclose()
        except Exception as e:
            logger.error("Redis Streams publish failed: %s", e)

    async def consume(self, event_type: EventType) -> AsyncIterator[Event]:
        """Async generator — consume events from a Redis Stream."""
        import aioredis
        redis = await aioredis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379")
        )
        stream = f"llmops:events:{event_type.value}"
        last_id = "0"
        try:
            while True:
                messages = await redis.xread({stream: last_id}, block=1000, count=10)
                for _, entries in (messages or []):
                    for entry_id, data in entries:
                        yield Event.from_json(data[b"data"].decode())
                        last_id = entry_id
        finally:
            await redis.aclose()

    async def consume_group(
        self,
        event_type: EventType,
        group: str,
        consumer: str,
    ) -> None:
        """Consume from a Redis consumer group and dispatch to registered handlers."""
        import aioredis
        redis = await aioredis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379")
        )
        stream = f"llmops:events:{event_type.value}"
        try:
            await redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass  # group already exists
        try:
            while True:
                messages = await redis.xreadgroup(
                    group, consumer, {stream: ">"}, count=10, block=1000
                )
                for _, entries in (messages or []):
                    for msg_id, data in entries:
                        event = Event.from_json(data[b"data"].decode())
                        for handler in self._handlers.get(event_type, []):
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(event)
                                else:
                                    handler(event)
                            except Exception as e:
                                logger.error("Consumer handler error: %s", e)
                        await redis.xack(stream, group, msg_id)
                await asyncio.sleep(0.05)
        finally:
            await redis.aclose()


# ---------------------------------------------------------------------------
# Base Microservice
# ---------------------------------------------------------------------------

class BaseMicroservice:
    """
    Base class for all LLMOps microservices.
    Provides a FastAPI app with /health and /metrics endpoints.
    """

    def __init__(self, name: str, port: int = 8000, version: str = "1.0.0"):
        self.name = name
        self.port = port
        self.version = version
        self.app = FastAPI(title=f"LLMOps {name} Service", version=version)
        self.bus = EventBus()
        self._request_count = 0
        self._error_count = 0
        self._start_time = time.time()
        self._setup_health()

    def _setup_health(self) -> None:
        @self.app.get("/health")
        async def health():
            return {
                "service": self.name,
                "status": "ok",
                "version": self.version,
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

        @self.app.get("/metrics")
        async def metrics():
            return {
                "service": self.name,
                "requests_total": self._request_count,
                "errors_total": self._error_count,
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

    def run(self) -> None:
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


# ---------------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------------

class RetrievalService(BaseMicroservice):
    """
    Dedicated retrieval service: FAISS search + distributed shard aggregation.
    """

    def __init__(self):
        super().__init__("Retrieval", port=8001)
        self._retriever = None
        self._setup_routes()

    def _get_retriever(self):
        if self._retriever is None:
            try:
                from agents.retriever import Retriever
                self._retriever = Retriever()
            except Exception as e:
                logger.warning("Could not load Retriever: %s — using mock", e)
                self._retriever = _MockRetriever()
        return self._retriever

    def _setup_routes(self) -> None:
        class RetrieveRequest(BaseModel):
            query: str
            top_k: int = 10
            correlation_id: str = ""

        class RetrieveResponse(BaseModel):
            documents: List[Dict[str, Any]]
            latency_ms: float
            correlation_id: str

        @self.app.post("/retrieve", response_model=RetrieveResponse)
        async def retrieve(req: RetrieveRequest):
            self._request_count += 1
            t0 = time.time()
            try:
                retriever = self._get_retriever()
                docs = retriever.retrieve(req.query, top_k=req.top_k)
                latency = (time.time() - t0) * 1000
                await self.bus.publish(Event(
                    type=EventType.RETRIEVAL_COMPLETE,
                    payload={"doc_count": len(docs), "latency_ms": latency},
                    correlation_id=req.correlation_id,
                    source_service=self.name,
                ))
                return RetrieveResponse(
                    documents=docs,
                    latency_ms=latency,
                    correlation_id=req.correlation_id,
                )
            except Exception as e:
                self._error_count += 1
                await self.bus.publish(Event(
                    type=EventType.ERROR,
                    payload={"service": self.name, "error": str(e)},
                    correlation_id=req.correlation_id,
                    source_service=self.name,
                ))
                raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Reranker Service
# ---------------------------------------------------------------------------

class RerankerService(BaseMicroservice):
    def __init__(self):
        super().__init__("Reranker", port=8002)
        self._reranker = None
        self._setup_routes()

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from agents.reranker import Reranker
                self._reranker = Reranker()
            except Exception as e:
                logger.warning("Could not load Reranker: %s — using mock", e)
                self._reranker = _MockReranker()
        return self._reranker

    def _setup_routes(self) -> None:
        class RerankRequest(BaseModel):
            query: str
            documents: List[Dict[str, Any]]
            top_k: int = 5
            correlation_id: str = ""

        @self.app.post("/rerank")
        async def rerank(req: RerankRequest):
            self._request_count += 1
            t0 = time.time()
            try:
                reranker = self._get_reranker()
                docs = reranker.rerank(req.query, req.documents, top_k=req.top_k)
                latency = (time.time() - t0) * 1000
                await self.bus.publish(Event(
                    type=EventType.RERANK_COMPLETE,
                    payload={"top_k": len(docs), "latency_ms": latency},
                    correlation_id=req.correlation_id,
                    source_service=self.name,
                ))
                return {"documents": docs, "latency_ms": latency}
            except Exception as e:
                self._error_count += 1
                raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Safety Service
# ---------------------------------------------------------------------------

class SafetyService(BaseMicroservice):
    """
    Dedicated safety screening service.
    Runs before retrieval/generation to block adversarial inputs.
    """

    def __init__(self):
        super().__init__("Safety", port=8004)
        self._checker = None
        self._setup_routes()

    def _get_checker(self):
        if self._checker is None:
            try:
                from safety.semantic_safety import SemanticSafetyChecker
                self._checker = SemanticSafetyChecker()
            except Exception as e:
                logger.warning("Could not load SafetyChecker: %s — using mock", e)
                self._checker = _MockSafetyChecker()
        return self._checker

    def _setup_routes(self) -> None:
        class SafetyRequest(BaseModel):
            text: str
            correlation_id: str = ""

        @self.app.post("/screen")
        async def screen(req: SafetyRequest):
            self._request_count += 1
            try:
                result = self._get_checker().check(req.text)
                event_type = (
                    EventType.SAFETY_CHECK_PASSED
                    if result.get("safe", True)
                    else EventType.SAFETY_FLAGGED
                )
                await self.bus.publish(Event(
                    type=event_type,
                    payload={"safe": result.get("safe", True),
                             "reason": result.get("reason", "")},
                    correlation_id=req.correlation_id,
                    source_service=self.name,
                ))
                return result
            except Exception as e:
                self._error_count += 1
                raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API Gateway
# ---------------------------------------------------------------------------

class APIGateway(BaseMicroservice):
    """
    Central gateway — orchestrates safety → retrieve → rerank → synthesize.
    Replaces the monolithic FastAPI app with a proper gateway pattern.
    """

    def __init__(self):
        super().__init__("Gateway", port=8000)
        self._setup_routes()

    def _setup_routes(self) -> None:
        class QueryRequest(BaseModel):
            query: str
            top_k: int = 5
            stream: bool = False

        @self.app.post("/query")
        async def query(req: QueryRequest):
            self._request_count += 1
            correlation_id = str(uuid.uuid4())

            await self.bus.publish(Event(
                type=EventType.QUERY_RECEIVED,
                payload={"query": req.query},
                correlation_id=correlation_id,
                source_service=self.name,
            ))

            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Safety screen
                try:
                    safety_resp = await client.post(
                        f"{ServiceRegistry.get('safety')}/screen",
                        json={"text": req.query, "correlation_id": correlation_id},
                    )
                    safety_result = safety_resp.json()
                    if not safety_result.get("safe", True):
                        return {
                            "error": "Query blocked by safety filter",
                            "reason": safety_result.get("reason"),
                        }
                except httpx.ConnectError:
                    logger.warning("Safety service unavailable — proceeding without check")

                # 2. Retrieve
                try:
                    retrieve_resp = await client.post(
                        f"{ServiceRegistry.get('retrieval')}/retrieve",
                        json={
                            "query": req.query,
                            "top_k": req.top_k * 5,
                            "correlation_id": correlation_id,
                        },
                    )
                    docs = retrieve_resp.json().get("documents", [])
                except httpx.ConnectError:
                    logger.warning("Retrieval service unavailable")
                    docs = []

                # 3. Rerank
                if docs:
                    try:
                        rerank_resp = await client.post(
                            f"{ServiceRegistry.get('reranker')}/rerank",
                            json={
                                "query": req.query,
                                "documents": docs,
                                "top_k": req.top_k,
                                "correlation_id": correlation_id,
                            },
                        )
                        reranked = rerank_resp.json().get("documents", docs[:req.top_k])
                    except httpx.ConnectError:
                        logger.warning("Reranker service unavailable — using raw results")
                        reranked = docs[:req.top_k]
                else:
                    reranked = []

                # 4. Synthesize
                try:
                    synth_resp = await client.post(
                        f"{ServiceRegistry.get('synthesizer')}/synthesize",
                        json={
                            "query": req.query,
                            "documents": reranked,
                            "correlation_id": correlation_id,
                        },
                    )
                    answer = synth_resp.json().get("answer", "")
                except httpx.ConnectError:
                    logger.warning("Synthesizer service unavailable")
                    answer = ""

            return {
                "answer": answer,
                "sources": reranked,
                "correlation_id": correlation_id,
            }

        @self.app.get("/services")
        async def list_services():
            return ServiceRegistry.list_services()

        @self.app.get("/services/health")
        async def services_health():
            return await ServiceRegistry.health_check_all()


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class _MockRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        return [{"text": f"Mock result {i} for: {query}", "score": 1.0 - i * 0.1}
                for i in range(top_k)]


class _MockReranker:
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


class _MockSafetyChecker:
    def check(self, text: str) -> Dict:
        return {"safe": True, "score": 0.95, "flags": [], "reason": ""}


# ---------------------------------------------------------------------------
# Service factories
# ---------------------------------------------------------------------------

def create_retrieval_service() -> FastAPI:
    return RetrievalService().app


def create_reranker_service() -> FastAPI:
    return RerankerService().app


def create_safety_service() -> FastAPI:
    return SafetyService().app


def create_gateway() -> FastAPI:
    return APIGateway().app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    service = os.environ.get("SERVICE", "gateway")
    port = int(os.environ.get("PORT", "8000"))

    factories = {
        "gateway":   create_gateway,
        "retrieval": create_retrieval_service,
        "reranker":  create_reranker_service,
        "safety":    create_safety_service,
    }
    app = factories.get(service, create_gateway)()
    uvicorn.run(app, host="0.0.0.0", port=port)
