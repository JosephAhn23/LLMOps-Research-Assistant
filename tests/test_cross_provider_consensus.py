"""Tests for cross-provider truth committee consensus (no live API calls)."""
from __future__ import annotations

import concurrent.futures

import pytest

from agents.multi_agent.cross_provider_consensus import (
    CrossProviderConsensusNode,
    LLMProvider,
    ProviderAnswer,
    text_agreement_score,
)
from agents.multi_agent.base_agent import AgentStatus


class _StubProvider(LLMProvider):
    def __init__(self, pid: str, text: str, err: str | None = None):
        self.provider_id = pid
        self.model = "stub"
        self._text = text
        self._err = err

    def complete(self, system: str, user: str) -> ProviderAnswer:
        if self._err:
            return ProviderAnswer(self.provider_id, self.model, "", error=self._err)
        return ProviderAnswer(self.provider_id, self.model, self._text, latency_ms=1.0)


class _DynamicStub(LLMProvider):
    """Returns different text based on prompt (for parallel calls)."""

    def __init__(self, pid: str, mapping: dict[str, str]):
        self.provider_id = pid
        self.model = "stub"
        self._mapping = mapping

    def complete(self, system: str, user: str) -> ProviderAnswer:
        return ProviderAnswer(
            self.provider_id, self.model, self._mapping.get(user, ""), latency_ms=0.5
        )


def test_text_agreement_identical():
    assert text_agreement_score("E = mc^2", "E = mc^2") == pytest.approx(1.0)


def test_text_agreement_different():
    s = text_agreement_score("The answer is 42.", "The answer is 41.")
    assert 0.0 < s < 1.0


def test_consensus_agrees_when_outputs_match():
    a = _StubProvider("p1", "The capital of France is Paris.")
    b = _StubProvider("p2", "The capital of France is Paris.")
    node = CrossProviderConsensusNode(a, b, agreement_threshold=0.8)
    r = node.run("Where is Paris?")
    assert r.models_agree is True
    assert r.hitl_required is False
    assert "Paris" in r.final_text


def test_consensus_hitl_when_outputs_diverge():
    a = _StubProvider("p1", "Result: 42")
    b = _StubProvider("p2", "Result: 17")
    node = CrossProviderConsensusNode(a, b, agreement_threshold=0.95)
    r = node.run("Compute?")
    assert r.models_agree is False
    assert r.hitl_required is True
    assert r.final_text == ""


def test_consensus_judge_overrides_low_similarity():
    a = _StubProvider("p1", "x = 2")
    b = _StubProvider("p2", "x equals two")

    def judge(x: str, y: str, _p: str) -> tuple[bool, str]:
        return True, "same meaning"

    node = CrossProviderConsensusNode(a, b, agreement_threshold=0.99, judge=judge)
    r = node.run("Solve")
    assert r.models_agree is True
    assert r.strategy == "llm_judge"
    assert r.hitl_required is False


def test_consensus_both_failed():
    a = _StubProvider("p1", "", err="timeout")
    b = _StubProvider("p2", "", err="rate limit")
    node = CrossProviderConsensusNode(a, b)
    r = node.run("q")
    assert r.hitl_required is True
    assert r.strategy == "both_failed"


def test_consensus_one_provider_fails():
    a = _StubProvider("p1", "only answer")
    b = _StubProvider("p2", "", err="down")
    node = CrossProviderConsensusNode(a, b)
    r = node.run("q")
    assert r.hitl_required is True
    assert r.strategy == "single_provider_fallback"
    assert r.final_text == "only answer"


def test_to_agent_results_maps_status():
    a = _StubProvider("p1", "ok")
    b = _StubProvider("p2", "ok")
    node = CrossProviderConsensusNode(a, b, agreement_threshold=0.5)
    r = node.run("q")
    agents = r.to_agent_results()
    assert len(agents) == 2
    assert all(ar.status == AgentStatus.SUCCEEDED for ar in agents)


def test_custom_executor():
    a = _DynamicStub("p1", {"q": "same"})
    b = _DynamicStub("p2", {"q": "same"})
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        node = CrossProviderConsensusNode(a, b, agreement_threshold=0.9, executor=ex)
        r = node.run("q")
    assert r.models_agree is True
