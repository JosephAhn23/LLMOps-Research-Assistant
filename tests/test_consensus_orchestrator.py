"""Adversarial consensus orchestrator — numeric verifier + mocked LLM path."""

from unittest.mock import MagicMock

from agents.multi_agent.consensus_orchestrator import (
    ConsensusOrchestrator,
    extract_floats,
    numeric_relative_conflict,
)


def test_extract_floats_basic() -> None:
    assert extract_floats("x = 3.14 and y = 2e-3") == [3.14, 0.002]


def test_numeric_relative_conflict_agrees() -> None:
    bad, msg = numeric_relative_conflict("result 100.0 units", "equals 100.0", relative_tolerance=0.001)
    assert bad is False
    assert msg == ""


def test_numeric_relative_conflict_detects_mismatch() -> None:
    bad, msg = numeric_relative_conflict("a 100.0", "b 100.2", relative_tolerance=0.001)
    assert bad is True
    assert "100" in msg


def test_consensus_orchestrator_happy_path() -> None:
    p1 = MagicMock()
    p2 = MagicMock()
    sk = MagicMock()
    p1.complete.return_value.text = "The value is 42.0 per [source_1]."
    p2.complete.return_value.text = "Answer: 42.0"
    sk.complete.return_value.text = "VERDICT: SOUND — consistent with context."

    orch = ConsensusOrchestrator(p1, p2, sk, max_hard_resets=1)
    out = orch.run("What is x?", "Context says x=42.")
    assert "42" in out.final_answer
    assert not out.hitl_recommended
    assert p1.complete.call_count >= 1


def test_consensus_orchestrator_numeric_conflict_then_reset() -> None:
    p1 = MagicMock()
    p2 = MagicMock()
    sk = MagicMock()
    p1.complete.side_effect = [
        MagicMock(text="Value 10.0"),
        MagicMock(text="Value 10.0"),
    ]
    p2.complete.side_effect = [
        MagicMock(text="Value 10.5"),
        MagicMock(text="Value 10.0"),
    ]
    sk.complete.return_value.text = "VERDICT: SOUND"

    orch = ConsensusOrchestrator(p1, p2, sk, max_hard_resets=2, numeric_tolerance=0.001)
    out = orch.run("q", "ctx")
    assert "10" in out.final_answer or "consensus" in out.final_answer.lower()
    assert len(out.conflict_events) >= 1
