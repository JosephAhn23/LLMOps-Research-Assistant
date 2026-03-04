"""
Base agent interface and typed state for the multi-agent system.

Every agent in the system inherits from BaseAgent and implements:
  - process(task: AgentTask) -> AgentResult
  - health() -> AgentHealth

Typed state prevents the "dict soup" problem common in multi-agent systems.
Structured logging ensures every agent action is traceable in production.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CIRCUIT_OPEN = "circuit_open"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    parent_task_id: Optional[str] = None
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    status: AgentStatus
    output: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        return self.status == AgentStatus.SUCCEEDED

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent_name,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "latency_ms": round(self.latency_ms, 1),
            "output_len": len(self.output),
            "error": self.error,
        }


@dataclass
class AgentHealth:
    agent_name: str
    status: AgentStatus
    total_tasks: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    circuit_open: bool = False
    last_error: Optional[str] = None


class ToolRegistry:
    """Registry of callable tools available to agents."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, name: str, fn: Callable, schema: Optional[Dict] = None) -> None:
        self._tools[name] = fn
        self._schemas[name] = schema or {"name": name, "description": fn.__doc__ or ""}
        logger.debug("Registered tool: %s", name)

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered.")
        return self._tools[name](**kwargs)

    def list_tools(self) -> List[Dict]:
        return list(self._schemas.values())


class BaseAgent:
    """
    Abstract base class for all agents in the multi-agent system.

    Provides:
    - Structured logging with task_id correlation
    - Latency measurement
    - Tool registry access
    - Health tracking
    """

    def __init__(
        self,
        name: str,
        tools: Optional[ToolRegistry] = None,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.tools = tools or ToolRegistry()
        self.timeout_seconds = timeout_seconds
        self._total_tasks = 0
        self._successes = 0
        self._latencies: List[float] = []
        self._last_error: Optional[str] = None
        self.logger = logging.getLogger(f"agent.{name}")

    def process(self, task: AgentTask) -> AgentResult:
        """Override in subclasses. Must return AgentResult."""
        raise NotImplementedError(f"{self.name}.process() not implemented.")

    def run(self, task: AgentTask) -> AgentResult:
        """Wrapper: measures latency, updates health stats, structured logging."""
        start = time.perf_counter()
        self._total_tasks += 1

        self.logger.info(
            "Starting task",
            extra={"task_id": task.task_id, "agent": self.name, "query_len": len(task.query)},
        )

        try:
            result = self.process(task)
            result.latency_ms = (time.perf_counter() - start) * 1000
            if result.is_success():
                self._successes += 1
            self._latencies.append(result.latency_ms)
            self.logger.info("Task complete", extra=result.to_log_dict())
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self._last_error = str(e)
            self.logger.error(
                "Task failed: %s",
                e,
                extra={"task_id": task.task_id, "agent": self.name},
                exc_info=True,
            )
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error=str(e),
                latency_ms=latency_ms,
            )

    def health(self) -> AgentHealth:
        return AgentHealth(
            agent_name=self.name,
            status=AgentStatus.IDLE,
            total_tasks=self._total_tasks,
            success_rate=self._successes / max(self._total_tasks, 1),
            avg_latency_ms=sum(self._latencies[-100:]) / max(len(self._latencies[-100:]), 1),
            last_error=self._last_error,
        )
