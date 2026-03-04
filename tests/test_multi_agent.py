"""
Comprehensive tests for the multi-agent system.

Tests cover:
  - Unit: each agent in isolation
  - Integration: full supervisor pipeline
  - Failure handling: circuit breaker, retry, graceful degradation
  - Consensus: all three strategies
  - Memory: TTL eviction, version conflicts, semantic retrieval
  - Routing: complexity classification, capability matching
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.multi_agent.base_agent import AgentResult, AgentStatus, AgentTask, ToolRegistry
from agents.multi_agent.consensus import (
    ConsensusResult,
    DebateRefinement,
    MajorityVote,
    WeightedConfidence,
    select_consensus_strategy,
)
from agents.multi_agent.critic_agent import CriticAgent
from agents.multi_agent.failure_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    GracefulDegradation,
    RetryConfig,
    RetryPolicy,
    TimeoutGuard,
)
from agents.multi_agent.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from agents.multi_agent.research_agent import ResearchAgent
from agents.multi_agent.routing import (
    CapabilityRouter,
    ComplexityRouter,
    classify_complexity,
    AgentCapability,
)
from agents.multi_agent.supervisor import Supervisor
from agents.multi_agent.verifier_agent import VerifierAgent


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_task():
    return AgentTask(query="What is RAG?", timeout_seconds=10.0)


@pytest.fixture
def complex_task():
    return AgentTask(
        query="Compare and analyze the tradeoffs between RAG and fine-tuning for domain adaptation.",
        timeout_seconds=30.0,
    )


@pytest.fixture
def researcher():
    return ResearchAgent()


@pytest.fixture
def critic():
    return CriticAgent(quality_threshold=0.70)


@pytest.fixture
def verifier():
    return VerifierAgent()


@pytest.fixture
def supervisor():
    return Supervisor(quality_threshold=0.70, max_iterations=2)


# ── Base Agent ─────────────────────────────────────────────────────────────────

class TestBaseAgent:
    def test_tool_registry_register_and_call(self):
        registry = ToolRegistry()
        registry.register("add", lambda a, b: a + b)
        result = registry.call("add", a=2, b=3)
        assert result == 5

    def test_tool_registry_missing_tool(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.call("nonexistent")

    def test_agent_health_initial(self, researcher):
        fresh = ResearchAgent()
        health = fresh.health()
        assert health.total_tasks == 0
        assert health.success_rate in (0.0, 1.0)

    def test_agent_run_updates_health(self, researcher, simple_task):
        researcher.run(simple_task)
        health = researcher.health()
        assert health.total_tasks == 1


# ── Research Agent ─────────────────────────────────────────────────────────────

class TestResearchAgent:
    def test_process_returns_result(self, researcher, simple_task):
        result = researcher.process(simple_task)
        assert result.status == AgentStatus.SUCCEEDED
        assert len(result.output) > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_process_with_critique_context(self, researcher):
        task = AgentTask(
            query="What is RAG?",
            context={"critique": "Answer is too brief. Add more detail."},
        )
        result = researcher.process(task)
        assert result.status == AgentStatus.SUCCEEDED

    def test_custom_retrieval_fn(self):
        custom_chunks = [{"doc_id": "custom_1", "text": "Custom retrieved text.", "score": 0.95}]
        agent = ResearchAgent(retrieval_fn=lambda **kw: custom_chunks)
        task = AgentTask(query="test query")
        result = agent.process(task)
        assert result.status == AgentStatus.SUCCEEDED
        assert "custom_1" in result.sources

    def test_confidence_low_without_context(self):
        agent = ResearchAgent(retrieval_fn=lambda **kw: [])
        task = AgentTask(query="test")
        result = agent.process(task)
        assert result.confidence <= 0.5

    def test_tool_calls_logged(self, researcher, simple_task):
        result = researcher.process(simple_task)
        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "retrieve" in tool_names
        assert "synthesize" in tool_names


# ── Critic Agent ───────────────────────────────────────────────────────────────

class TestCriticAgent:
    def test_critique_with_good_answer(self, critic):
        task = AgentTask(
            query="What is RAG?",
            context={
                "research_result": (
                    "RAG (Retrieval-Augmented Generation) typically uses a retrieval step "
                    "to ground LLM responses in external knowledge. This may improve factual "
                    "accuracy and can reduce hallucinations in many cases."
                ),
                "retrieved_context": [
                    {"text": "RAG uses retrieval to ground LLM responses in external knowledge.", "score": 0.9},
                    {"text": "Retrieval-augmented generation reduces hallucinations.", "score": 0.85},
                ],
            },
        )
        result = critic.process(task)
        assert result.status == AgentStatus.SUCCEEDED
        assert result.confidence > 0.0
        assert "overall_score" in result.metadata

    def test_critique_fails_without_research_result(self, critic):
        task = AgentTask(query="test", context={})
        result = critic.process(task)
        assert result.status == AgentStatus.FAILED

    def test_critique_identifies_issues(self, critic):
        task = AgentTask(
            query="Explain fine-tuning in detail with examples",
            context={
                "research_result": "Fine-tuning is good.",
                "retrieved_context": [],
            },
        )
        result = critic.process(task)
        assert result.status == AgentStatus.SUCCEEDED
        assert len(result.metadata.get("issues", [])) > 0

    def test_needs_revision_below_threshold(self, critic):
        task = AgentTask(
            query="Complex question",
            context={"research_result": "Short.", "retrieved_context": []},
        )
        result = critic.process(task)
        assert result.metadata.get("needs_revision") is True


# ── Verifier Agent ─────────────────────────────────────────────────────────────

class TestVerifierAgent:
    def test_verify_supported_claims(self, verifier):
        task = AgentTask(
            query="What is RAG?",
            context={
                "research_result": "RAG uses retrieval to ground LLM responses in external knowledge.",
                "retrieved_context": [
                    {"text": "RAG uses retrieval to ground LLM responses in external knowledge.", "score": 0.95},
                ],
            },
        )
        result = verifier.process(task)
        assert result.status == AgentStatus.SUCCEEDED
        verifications = result.metadata.get("verifications", [])
        assert len(verifications) > 0

    def test_verify_fails_without_answer(self, verifier):
        task = AgentTask(query="test", context={})
        result = verifier.process(task)
        assert result.status == AgentStatus.FAILED

    def test_veracity_score_range(self, verifier):
        task = AgentTask(
            query="test",
            context={
                "research_result": "This is a claim. Another claim here. Third claim.",
                "retrieved_context": [{"text": "This is a claim with supporting evidence.", "score": 0.8}],
            },
        )
        result = verifier.process(task)
        assert 0.0 <= result.metadata.get("veracity_score", 0.0) <= 1.0


# ── Consensus ──────────────────────────────────────────────────────────────────

class TestConsensus:
    def _make_results(self, outputs_and_confidences):
        return [
            AgentResult(
                task_id="t1",
                agent_name=f"agent_{i}",
                status=AgentStatus.SUCCEEDED,
                output=out,
                confidence=conf,
            )
            for i, (out, conf) in enumerate(outputs_and_confidences)
        ]

    def test_majority_vote_clear_winner(self):
        results = self._make_results([
            ("answer A", 0.8),
            ("answer A", 0.7),
            ("answer B", 0.9),
        ])
        consensus = MajorityVote().aggregate(results)
        assert "answer a" in consensus.final_output.lower()
        assert consensus.agreement_score == pytest.approx(2 / 3, abs=0.01)

    def test_majority_vote_empty(self):
        consensus = MajorityVote().aggregate([])
        assert consensus.confidence == 0.0

    def test_weighted_confidence_selects_highest(self):
        results = self._make_results([
            ("low confidence answer", 0.3),
            ("high confidence answer", 0.95),
        ])
        consensus = WeightedConfidence().aggregate(results)
        assert "high confidence" in consensus.final_output

    def test_weighted_confidence_with_agent_weights(self):
        results = self._make_results([
            ("answer from trusted agent", 0.7),
            ("answer from untrusted agent", 0.9),
        ])
        consensus = WeightedConfidence(agent_weights={"agent_0": 2.0, "agent_1": 0.5}).aggregate(results)
        assert "trusted" in consensus.final_output

    def test_debate_refinement_runs(self):
        results = self._make_results([("answer 1", 0.7), ("answer 2", 0.65)])
        consensus = DebateRefinement(max_rounds=2).aggregate(results)
        assert consensus.rounds >= 1
        assert len(consensus.final_output) > 0

    def test_select_strategy_categorical(self):
        results = self._make_results([("yes", 0.8), ("no", 0.7)])
        strategy = select_consensus_strategy(results, complexity="low")
        assert strategy == "majority_vote"

    def test_select_strategy_high_complexity(self):
        results = self._make_results([("long detailed answer " * 5, 0.8)] * 3)
        strategy = select_consensus_strategy(results, complexity="high", latency_budget_ms=5000)
        assert strategy == "debate"


# ── Memory ─────────────────────────────────────────────────────────────────────

class TestMemory:
    def test_short_term_write_read(self):
        mem = ShortTermMemory()
        mem.write("key1", "value1", author="agent_a")
        val, version = mem.read("key1")
        assert val == "value1"
        assert version == 1

    def test_short_term_version_increment(self):
        mem = ShortTermMemory()
        mem.write("key1", "v1", author="a")
        mem.write("key1", "v2", author="b")
        val, version = mem.read("key1")
        assert val == "v2"
        assert version == 2

    def test_short_term_version_conflict(self):
        mem = ShortTermMemory()
        mem.write("key1", "v1", author="a")
        with pytest.raises(ValueError, match="Version conflict"):
            mem.write("key1", "v2", author="b", expected_version=0)

    def test_short_term_ttl_eviction(self):
        mem = ShortTermMemory(default_ttl_seconds=0.01)
        mem.write("key1", "value", author="a")
        time.sleep(0.05)
        val, version = mem.read("key1")
        assert val is None
        assert version == 0

    def test_long_term_store_and_retrieve(self):
        ltm = LongTermMemory()
        ltm.store("fact1", "RAG uses retrieval to ground responses.")
        ltm.store("fact2", "Fine-tuning updates model weights.")
        results = ltm.retrieve("retrieval augmented generation", top_k=2)
        assert len(results) <= 2
        assert all("text" in r for r in results)

    def test_working_memory_set_get(self):
        wm = WorkingMemory("session_1")
        wm.set("answer", "test answer", author="researcher")
        val, version = wm.get("answer")
        assert val == "test answer"
        assert version == 1

    def test_working_memory_recall(self):
        wm = WorkingMemory("session_2")
        wm.remember("fact1", "RAG is a retrieval method.")
        results = wm.recall("retrieval method", top_k=1)
        assert len(results) == 1


# ── Failure Handling ───────────────────────────────────────────────────────────

class TestFailureHandling:
    def test_circuit_breaker_opens_after_threshold(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            cb.record_failure()
        assert cb.state.value == "open"
        assert not cb.allow_call()

    def test_circuit_breaker_half_open_after_timeout(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.01))
        cb.record_failure()
        time.sleep(0.05)
        assert cb.state.value == "half_open"
        assert cb.allow_call()

    def test_circuit_breaker_closes_after_success(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.01, success_threshold=1))
        cb.record_failure()
        time.sleep(0.05)
        cb.allow_call()
        cb.record_success()
        assert cb.state.value == "closed"

    def test_retry_policy_succeeds_on_third_attempt(self):
        attempts = [0]
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Transient error")
            return "success"

        policy = RetryPolicy(RetryConfig(max_attempts=3, base_delay_seconds=0.001))
        result = policy.execute(flaky, context="test")
        assert result == "success"
        assert attempts[0] == 3

    def test_retry_policy_raises_after_max_attempts(self):
        policy = RetryPolicy(RetryConfig(max_attempts=2, base_delay_seconds=0.001))
        with pytest.raises(RuntimeError, match="All 2 attempts failed"):
            policy.execute(lambda: (_ for _ in ()).throw(ValueError("always fails")), context="test")

    def test_graceful_degradation_uses_fallback(self):
        def primary(*args, **kwargs):
            raise RuntimeError("primary down")

        def fallback(*args, **kwargs):
            return "fallback result"

        chain = GracefulDegradation([("primary", primary), ("fallback", fallback)])
        result, provider, confidence = chain.execute()
        assert result == "fallback result"
        assert provider == "fallback"
        assert confidence < 1.0

    def test_timeout_guard_raises(self):
        guard = TimeoutGuard(timeout_seconds=0.001)
        with guard:
            time.sleep(0.05)
            with pytest.raises(TimeoutError):
                guard.check()


# ── Routing ────────────────────────────────────────────────────────────────────

class TestRouting:
    def test_classify_complexity_high(self):
        assert classify_complexity("Compare and analyze the tradeoffs between RAG and fine-tuning") == "high"

    def test_classify_complexity_low(self):
        assert classify_complexity("What is RAG?") == "low"

    def test_classify_complexity_medium(self):
        assert classify_complexity("How does RAG work in practice?") == "medium"

    def test_complexity_router_high_uses_all_agents(self):
        router = ComplexityRouter()
        task = AgentTask(query="Compare and analyze the tradeoffs between RAG and fine-tuning")
        decision = router.route(task, ["researcher", "critic", "verifier"])
        assert len(decision.selected_agents) == 3

    def test_complexity_router_low_uses_one_agent(self):
        router = ComplexityRouter()
        task = AgentTask(query="What is RAG?")
        decision = router.route(task, ["researcher", "critic", "verifier"])
        assert len(decision.selected_agents) == 1

    def test_capability_router_finds_matching_agent(self):
        caps = {
            "researcher": AgentCapability("researcher", {"retrieval", "reasoning"}),
            "coder": AgentCapability("coder", {"code_execution", "reasoning"}),
        }
        router = CapabilityRouter(caps)
        task = AgentTask(query="find documents about RAG")
        decision = router.route(task, required_capabilities={"retrieval"})
        assert "researcher" in decision.selected_agents


# ── Integration: Full Supervisor Pipeline ─────────────────────────────────────

class TestSupervisorIntegration:
    def test_simple_query_returns_answer(self, supervisor, simple_task):
        trace = supervisor.run(simple_task.query)
        assert len(trace.final_answer) > 0
        assert trace.total_latency_ms > 0
        assert trace.iterations >= 1

    def test_complex_query_runs_multiple_iterations(self, supervisor):
        trace = supervisor.run(
            "Compare and analyze the tradeoffs between RAG and fine-tuning for domain adaptation."
        )
        assert trace.iterations >= 1
        assert trace.consensus is not None

    def test_supervisor_health_returns_all_agents(self, supervisor):
        health = supervisor.health()
        assert "researcher" in health["agents"]
        assert "critic" in health["agents"]
        assert "verifier" in health["agents"]

    def test_hitl_triggered_on_low_confidence(self):
        # Force HITL by setting hitl_threshold above any possible confidence
        sup = Supervisor(quality_threshold=0.99, hitl_threshold=2.0, max_iterations=1)
        trace = sup.run("What is RAG?")
        assert trace.hitl_triggered

    def test_hitl_resolve(self, supervisor):
        trace = supervisor.run("What is RAG?")
        if trace.hitl_triggered:
            resolved = supervisor.resolve_hitl(trace.session_id, "Human-reviewed answer.")
            assert resolved

    def test_circuit_breaker_status_in_health(self, supervisor):
        health = supervisor.health()
        for name in ["researcher", "critic", "verifier"]:
            assert name in health["circuit_breakers"]
            assert health["circuit_breakers"][name]["state"] in ["closed", "open", "half_open"]

    def test_agent_results_in_trace(self, supervisor, simple_task):
        trace = supervisor.run(simple_task.query)
        agent_names = [r.agent_name for r in trace.agent_results]
        assert "researcher" in agent_names

    def test_session_id_propagated(self, supervisor):
        trace = supervisor.run("What is RAG?", session_id="test_session_123")
        assert trace.session_id == "test_session_123"
