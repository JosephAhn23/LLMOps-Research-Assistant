"""Policy enforcement: drop unsupported sentences."""

from agents.multi_agent.policy_enforcement_agent import (
    PolicyEnforcementAgent,
    enforce_grounded_answer,
)
from agents.multi_agent.base_agent import AgentStatus, AgentTask


def test_enforce_keeps_cited_sentence() -> None:
    chunks = [{"text": "The refund deadline is fourteen days.", "source": "policy.md"}]
    ans = "Per policy, the refund deadline is fourteen days [source_1]."
    r = enforce_grounded_answer(ans, chunks)
    assert "fourteen days" in r.enforced_answer
    assert not r.dropped_sentences


def test_enforce_drops_unsupported_claim() -> None:
    chunks = [{"text": "Widgets are blue.", "source": "doc.md"}]
    ans = (
        "Widgets are blue. The company unconditionally guarantees lifetime free "
        "hardware replacements for every registered customer worldwide without exception."
    )
    r = enforce_grounded_answer(ans, chunks)
    assert "blue" in r.enforced_answer
    assert r.dropped_sentences
    assert any("replacements" in d or "guarantees" in d for d in r.dropped_sentences)


def test_policy_agent_process() -> None:
    agent = PolicyEnforcementAgent()
    task = AgentTask(
        task_id="t1",
        query="q",
        context={
            "draft_answer": "Alpha [source_1]. Beta is always true everywhere.",
            "retrieved_context": [{"text": "Alpha detail here.", "source": "a.md"}],
        },
    )
    res = agent.process(task)
    assert res.status == AgentStatus.SUCCEEDED
    assert "Alpha" in (res.metadata or {}).get("enforced_answer", "")
