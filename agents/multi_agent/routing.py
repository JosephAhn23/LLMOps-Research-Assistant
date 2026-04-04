"""
Dynamic task routing for the multi-agent supervisor.

Routing strategies:
  1. ComplexityRouter: routes by estimated query complexity
  2. CapabilityRouter: routes to agents with matching capability tags
  3. PerformanceRouter: routes to historically best-performing agent for task type
  4. LoadBalancer: round-robin with health-aware exclusion

The supervisor uses these routers to decide which agents to invoke
and in what configuration (parallel, sequential, or ensemble).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from agents.multi_agent.base_agent import AgentHealth, AgentStatus, AgentTask, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Describes what an agent can do."""
    name: str
    capabilities: Set[str]
    max_concurrent: int = 1
    avg_latency_ms: float = 1000.0
    quality_score: float = 0.8
    cost_per_call: float = 0.001


@dataclass
class RoutingDecision:
    selected_agents: List[str]
    strategy: str
    reasoning: str
    parallel: bool = True
    consensus_strategy: str = "weighted_confidence"
    estimated_latency_ms: float = 0.0


COMPLEXITY_SIGNALS = {
    "high": [
        "compare", "analyze", "critique", "design", "tradeoff", "evaluate",
        "explain why", "what are the implications", "how would you", "debate",
        "argue", "synthesize", "multi-step",
    ],
    "low": [
        "what is", "define", "list", "when was", "who is", "how many",
        "yes or no", "true or false",
    ],
}


def classify_complexity(query: str) -> str:
    q = query.lower()
    high_hits = sum(1 for s in COMPLEXITY_SIGNALS["high"] if s in q)
    low_hits = sum(1 for s in COMPLEXITY_SIGNALS["low"] if s in q)
    if high_hits > low_hits:
        return "high"
    if low_hits > high_hits:
        return "low"
    return "medium"


class ComplexityRouter:
    """
    Routes tasks based on query complexity.

    Low complexity: single fast agent
    Medium complexity: research + verifier
    High complexity: full ensemble (research + critic + verifier)
    """

    ROUTING_TABLE = {
        "low": {
            "agents": ["researcher"],
            "parallel": False,
            "consensus": "weighted_confidence",
        },
        "medium": {
            "agents": ["researcher", "verifier"],
            "parallel": True,
            "consensus": "weighted_confidence",
        },
        "high": {
            "agents": ["researcher", "critic", "verifier"],
            "parallel": True,
            "consensus": "debate",
        },
    }

    def route(self, task: AgentTask, available_agents: List[str]) -> RoutingDecision:
        complexity = classify_complexity(task.query)
        config = self.ROUTING_TABLE.get(complexity, self.ROUTING_TABLE["medium"])

        selected = [a for a in config["agents"] if a in available_agents]
        if not selected:
            selected = available_agents[:1]

        return RoutingDecision(
            selected_agents=selected,
            strategy=f"complexity:{complexity}",
            reasoning=f"Query classified as '{complexity}' complexity.",
            parallel=config["parallel"],
            consensus_strategy=config["consensus"],
            estimated_latency_ms=1000.0 * len(selected) if not config["parallel"] else 1200.0,
        )


class CapabilityRouter:
    """
    Routes to agents that have the required capabilities for a task.

    Capabilities are declared by each agent (e.g., "retrieval", "reasoning",
    "fact_checking", "code_execution"). The router finds agents that cover
    all required capabilities.
    """

    def __init__(self, agent_capabilities: Dict[str, AgentCapability]):
        self.capabilities = agent_capabilities

    def route(
        self,
        task: AgentTask,
        required_capabilities: Optional[Set[str]] = None,
    ) -> RoutingDecision:
        if not required_capabilities:
            required_capabilities = self._infer_capabilities(task.query)

        matching = [
            name for name, cap in self.capabilities.items()
            if required_capabilities.issubset(cap.capabilities)
        ]

        if not matching:
            covering = self._find_covering_set(required_capabilities)
            return RoutingDecision(
                selected_agents=covering,
                strategy="capability:covering_set",
                reasoning=f"No single agent covers {required_capabilities}. Using covering set.",
                parallel=True,
                consensus_strategy="weighted_confidence",
            )

        best = min(matching, key=lambda n: self.capabilities[n].avg_latency_ms / self.capabilities[n].quality_score)
        return RoutingDecision(
            selected_agents=[best],
            strategy="capability:best_match",
            reasoning=f"Agent '{best}' covers all required capabilities: {required_capabilities}.",
            parallel=False,
            consensus_strategy="weighted_confidence",
        )

    def _infer_capabilities(self, query: str) -> Set[str]:
        caps = {"reasoning"}
        q = query.lower()
        if any(w in q for w in ["fact", "verify", "check", "true", "accurate"]):
            caps.add("fact_checking")
        if any(w in q for w in ["retrieve", "find", "search", "document", "source"]):
            caps.add("retrieval")
        if any(w in q for w in ["code", "implement", "function", "class", "python"]):
            caps.add("code_execution")
        return caps

    def _find_covering_set(self, required: Set[str]) -> List[str]:
        covered: Set[str] = set()
        selected = []
        remaining = set(required)
        sorted_agents = sorted(
            self.capabilities.items(),
            key=lambda x: len(x[1].capabilities & remaining),
            reverse=True,
        )
        for name, cap in sorted_agents:
            if not remaining:
                break
            new_coverage = cap.capabilities & remaining
            if new_coverage:
                selected.append(name)
                remaining -= new_coverage
        return selected


class PerformanceRouter:
    """
    Routes to the historically best-performing agent for a given task type.

    Tracks per-agent performance metrics (success rate, latency, quality)
    and routes to the agent with the best risk-adjusted score.

    Implements epsilon-greedy exploration: with probability epsilon,
    routes to a random agent to gather performance data on new agents.
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon
        self._performance: Dict[str, Dict[str, float]] = {}

    def record(self, agent_name: str, success: bool, latency_ms: float, quality: float = 0.8) -> None:
        if agent_name not in self._performance:
            self._performance[agent_name] = {
                "n": 0, "successes": 0, "total_latency": 0.0, "total_quality": 0.0,
            }
        p = self._performance[agent_name]
        p["n"] += 1
        p["successes"] += int(success)
        p["total_latency"] += latency_ms
        p["total_quality"] += quality

    def score(self, agent_name: str) -> float:
        p = self._performance.get(agent_name)
        if not p or p["n"] == 0:
            return 0.5
        success_rate = p["successes"] / p["n"]
        avg_latency = p["total_latency"] / p["n"]
        avg_quality = p["total_quality"] / p["n"]
        latency_score = max(0.0, 1.0 - avg_latency / 10000.0)
        return 0.5 * success_rate + 0.3 * avg_quality + 0.2 * latency_score

    def route(self, available_agents: List[str], task: AgentTask) -> RoutingDecision:
        import random
        if random.random() < self.epsilon:
            selected = random.choice(available_agents)
            return RoutingDecision(
                selected_agents=[selected],
                strategy="performance:explore",
                reasoning=f"Epsilon-greedy exploration: selected '{selected}'.",
                parallel=False,
                consensus_strategy="weighted_confidence",
            )

        best = max(available_agents, key=self.score)
        return RoutingDecision(
            selected_agents=[best],
            strategy="performance:exploit",
            reasoning=f"Best historical performer: '{best}' (score={self.score(best):.3f}).",
            parallel=False,
            consensus_strategy="weighted_confidence",
        )


class LoadBalancer:
    """
    Round-robin load balancer with health-aware exclusion.
    Skips agents with open circuit breakers or high error rates.
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._round_robin_idx = 0

    def select(
        self,
        available_agents: List[str],
        health_map: Optional[Dict[str, AgentHealth]] = None,
    ) -> str:
        healthy = available_agents
        if health_map:
            healthy = [
                a for a in available_agents
                if not health_map.get(a, AgentHealth(a, AgentStatus.IDLE)).circuit_open
                and health_map.get(a, AgentHealth(a, AgentStatus.IDLE)).success_rate > 0.5
            ]
        if not healthy:
            healthy = available_agents

        selected = healthy[self._round_robin_idx % len(healthy)]
        self._round_robin_idx += 1
        self._counters[selected] = self._counters.get(selected, 0) + 1
        return selected

    def stats(self) -> Dict[str, int]:
        return dict(self._counters)
