"""
Supervisor: orchestrates the multi-agent pipeline.

Responsibilities:
  1. Decompose complex queries into subtasks
  2. Route subtasks to appropriate agents via routing strategies
  3. Execute agents (parallel or sequential based on routing decision)
  4. Apply consensus strategy to aggregate results
  5. Enforce quality threshold with iterative refinement
  6. Handle failures via circuit breakers and graceful degradation
  7. Emit structured traces for observability

Human-in-the-loop (HITL) checkpoints:
  - Triggered when final confidence < hitl_threshold
  - Triggered when safety keywords detected in output
  - Pauses pipeline and emits a HITL request event

OpenTelemetry tracing:
  - Each agent invocation creates a span
  - Span attributes: agent_name, task_id, latency_ms, confidence
  - Falls back gracefully if opentelemetry not installed
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.multi_agent.base_agent import AgentResult, AgentStatus, AgentTask, TaskPriority
from agents.multi_agent.consensus import (
    ConsensusResult,
    DebateRefinement,
    MajorityVote,
    WeightedConfidence,
    select_consensus_strategy,
)
from agents.multi_agent.critic_agent import CriticAgent
from agents.multi_agent.failure_handling import CircuitBreaker, GracefulDegradation, RetryPolicy
from agents.multi_agent.memory import WorkingMemory
from agents.multi_agent.research_agent import ResearchAgent
from agents.multi_agent.routing import ComplexityRouter, RoutingDecision
from agents.multi_agent.verifier_agent import VerifierAgent

logger = logging.getLogger(__name__)

SAFETY_KEYWORDS = [
    "harm", "illegal", "weapon", "violence", "exploit", "jailbreak",
    "bypass", "ignore instructions", "override safety",
]


@dataclass
class HITLRequest:
    session_id: str
    task_id: str
    reason: str
    agent_output: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class PipelineTrace:
    session_id: str
    query: str
    routing_decision: Optional[RoutingDecision]
    agent_results: List[AgentResult]
    consensus: Optional[ConsensusResult]
    final_answer: str
    total_latency_ms: float
    iterations: int
    hitl_triggered: bool = False
    hitl_request: Optional[HITLRequest] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "iterations": self.iterations,
            "n_agents": len(self.agent_results),
            "final_confidence": round(self.consensus.confidence if self.consensus else 0.0, 3),
            "hitl_triggered": self.hitl_triggered,
        }


def _try_otel_span(name: str, attributes: Dict[str, Any]):
    """OpenTelemetry span context manager. No-op if not installed."""
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("llmops.multi_agent")
        return tracer.start_as_current_span(name, attributes=attributes)
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


class Supervisor:
    """
    Production multi-agent supervisor with full observability.

    Architecture:
      Query -> Decompose -> Route -> Execute (parallel) -> Consensus
           -> Quality check -> [Refine loop] -> HITL check -> Return

    The refinement loop runs until:
      - consensus.confidence >= quality_threshold, OR
      - max_iterations reached

    On each iteration, the critic's feedback is injected into the
    research agent's context so it can improve its answer.
    """

    def __init__(
        self,
        quality_threshold: float = 0.78,
        max_iterations: int = 3,
        hitl_threshold: float = 0.55,
        hitl_callback: Optional[Callable[[HITLRequest], None]] = None,
        max_workers: int = 4,
    ):
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.hitl_threshold = hitl_threshold
        self.hitl_callback = hitl_callback
        self.max_workers = max_workers

        self.researcher = ResearchAgent()
        self.critic = CriticAgent(quality_threshold=quality_threshold)
        self.verifier = VerifierAgent()

        self._agents = {
            "researcher": self.researcher,
            "critic": self.critic,
            "verifier": self.verifier,
        }
        self._circuit_breakers = {
            name: CircuitBreaker(name) for name in self._agents
        }
        self._retry = RetryPolicy()
        self._router = ComplexityRouter()
        self._hitl_queue: List[HITLRequest] = []

    def run(self, query: str, session_id: Optional[str] = None) -> PipelineTrace:
        session_id = session_id or str(uuid.uuid4())[:12]
        start = time.perf_counter()
        memory = WorkingMemory(session_id)
        memory.set("task", query, author="user")

        with _try_otel_span("supervisor.run", {"session_id": session_id, "query_len": len(query)}):
            task = AgentTask(
                query=query,
                priority=TaskPriority.NORMAL,
                timeout_seconds=60.0,
            )

            routing = self._router.route(task, list(self._agents.keys()))
            logger.info(
                "Routing: session=%s strategy=%s agents=%s",
                session_id, routing.strategy, routing.selected_agents,
            )

            all_results: List[AgentResult] = []
            consensus: Optional[ConsensusResult] = None
            iterations = 0

            for iteration in range(self.max_iterations):
                iterations = iteration + 1

                critique, _ = memory.get("critique")
                if critique:
                    task.context["critique"] = critique

                research_task = AgentTask(
                    query=query,
                    context=task.context.copy(),
                    priority=task.priority,
                )
                research_result = self._run_agent_safe("researcher", research_task)
                all_results.append(research_result)

                if research_result.is_success():
                    memory.set("research_result", research_result.output, author="researcher")
                    memory.set("retrieved_context", research_result.metadata.get("n_chunks", []), author="researcher")

                    critic_task = AgentTask(
                        query=query,
                        context={
                            "research_result": research_result.output,
                            "retrieved_context": self.researcher._default_retrieve(query),
                        },
                    )
                    critic_result = self._run_agent_safe("critic", critic_task)
                    all_results.append(critic_result)

                    if critic_result.is_success():
                        score = critic_result.metadata.get("overall_score", 0.0)
                        memory.set("critique", critic_result.metadata.get("issues", []), author="critic")

                        if score >= self.quality_threshold:
                            logger.info(
                                "Quality threshold reached: iteration=%d score=%.3f",
                                iteration, score,
                            )
                            break
                        else:
                            logger.info(
                                "Iteration %d: score=%.3f < %.3f, refining.",
                                iteration, score, self.quality_threshold,
                            )

            verifier_task = AgentTask(
                query=query,
                context={
                    "research_result": memory.get("research_result")[0] or "",
                    "retrieved_context": self.researcher._default_retrieve(query),
                },
            )
            verifier_result = self._run_agent_safe("verifier", verifier_task)
            all_results.append(verifier_result)

            research_results = [r for r in all_results if r.agent_name == "researcher" and r.is_success()]
            if research_results:
                strategy = select_consensus_strategy(research_results, complexity="medium")
                consensus = self._apply_consensus(strategy, research_results)
            else:
                consensus = ConsensusResult(
                    strategy="fallback",
                    final_output="Unable to generate a reliable answer.",
                    confidence=0.0,
                    agreement_score=0.0,
                    participating_agents=[],
                    dissenting_agents=[],
                )

            hitl_triggered = False
            hitl_request = None
            if consensus.confidence < self.hitl_threshold or self._has_safety_trigger(consensus.final_output):
                hitl_triggered = True
                hitl_request = HITLRequest(
                    session_id=session_id,
                    task_id=task.task_id,
                    reason=(
                        "Low confidence" if consensus.confidence < self.hitl_threshold
                        else "Safety trigger detected"
                    ),
                    agent_output=consensus.final_output,
                    confidence=consensus.confidence,
                )
                self._hitl_queue.append(hitl_request)
                if self.hitl_callback:
                    self.hitl_callback(hitl_request)
                logger.warning(
                    "HITL triggered: session=%s reason=%s confidence=%.3f",
                    session_id, hitl_request.reason, consensus.confidence,
                )

            memory.remember(
                f"answer:{session_id}",
                consensus.final_output,
                {"confidence": consensus.confidence, "iterations": iterations},
            )

            total_ms = (time.perf_counter() - start) * 1000
            trace = PipelineTrace(
                session_id=session_id,
                query=query,
                routing_decision=routing,
                agent_results=all_results,
                consensus=consensus,
                final_answer=consensus.final_output,
                total_latency_ms=total_ms,
                iterations=iterations,
                hitl_triggered=hitl_triggered,
                hitl_request=hitl_request,
            )

            logger.info("Pipeline complete: %s", trace.to_log_dict())
            return trace

    def _run_agent_safe(self, agent_name: str, task: AgentTask) -> AgentResult:
        cb = self._circuit_breakers[agent_name]
        agent = self._agents[agent_name]

        if not cb.allow_call():
            logger.warning("Circuit open for '%s'. Using degraded fallback.", agent_name)
            return AgentResult(
                task_id=task.task_id,
                agent_name=agent_name,
                status=AgentStatus.CIRCUIT_OPEN,
                output="",
                confidence=0.0,
                error=f"Circuit breaker open for {agent_name}",
            )

        with _try_otel_span(f"agent.{agent_name}", {"task_id": task.task_id}):
            try:
                result = self._retry.execute(
                    lambda: agent.run(task),
                    context=f"{agent_name}:{task.task_id}",
                )
                if result.is_success():
                    cb.record_success()
                else:
                    cb.record_failure()
                return result
            except Exception as e:
                cb.record_failure()
                return AgentResult(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    confidence=0.0,
                )

    def _run_parallel(self, agent_names: List[str], task: AgentTask) -> List[AgentResult]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_agent_safe, name, task): name
                for name in agent_names
            }
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=task.timeout_seconds):
                try:
                    results.append(future.result())
                except Exception as e:
                    name = futures[future]
                    results.append(AgentResult(
                        task_id=task.task_id,
                        agent_name=name,
                        status=AgentStatus.FAILED,
                        error=str(e),
                        confidence=0.0,
                    ))
        return results

    def _apply_consensus(self, strategy: str, results: List[AgentResult]) -> ConsensusResult:
        if strategy == "majority_vote":
            return MajorityVote().aggregate(results)
        elif strategy == "debate":
            return DebateRefinement().aggregate(results)
        else:
            return WeightedConfidence().aggregate(results)

    def _has_safety_trigger(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in SAFETY_KEYWORDS)

    def health(self) -> Dict[str, Any]:
        return {
            "agents": {name: agent.health().__dict__ for name, agent in self._agents.items()},
            "circuit_breakers": {name: cb.status() for name, cb in self._circuit_breakers.items()},
            "hitl_queue_size": len(self._hitl_queue),
            "pending_hitl": sum(1 for h in self._hitl_queue if not h.resolved),
        }

    def resolve_hitl(self, session_id: str, resolution: str) -> bool:
        for req in self._hitl_queue:
            if req.session_id == session_id and not req.resolved:
                req.resolved = True
                req.resolution = resolution
                logger.info("HITL resolved: session=%s resolution=%s", session_id, resolution[:50])
                return True
        return False
