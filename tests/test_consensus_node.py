"""Tests for ConsensusNode (mocked committee)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.multi_agent.consensus_node import ConsensusNode, ConsensusState
from agents.multi_agent.cross_provider_consensus import CrossProviderConsensusResult, ProviderAnswer


def _result(
    *,
    hitl: bool,
    strategy: str = "similarity",
    a: str = "a",
    b: str = "b",
) -> CrossProviderConsensusResult:
    pa = ProviderAnswer("p1", "m", a)
    pb = ProviderAnswer("p2", "m", b)
    agree = not hitl
    return CrossProviderConsensusResult(
        task_id="t",
        prompt_excerpt="",
        answers=(pa, pb),
        agreement_score=0.99 if agree else 0.1,
        models_agree=agree,
        final_text=a if agree else "",
        strategy=strategy,
        hitl_required=hitl,
        hitl_reason="" if agree else "disagree",
    )


def test_consensus_node_ok_first_round():
    primary = MagicMock()
    secondary = MagicMock()
    inner = MagicMock()
    inner.run.return_value = _result(hitl=False, a="final", b="final")
    with patch("agents.multi_agent.consensus_node.CrossProviderConsensusNode", return_value=inner):
        node = ConsensusNode(primary, secondary, use_openai_judge=False, max_self_corrections=2)
        out = node.run("prompt")
    assert out.state == ConsensusState.CONSENSUS_OK
    assert out.final_answer == "final"
    assert out.attempts == 1
    inner.run.assert_called_once()


def test_consensus_node_self_corrects_then_ok():
    primary = MagicMock()
    secondary = MagicMock()
    inner = MagicMock()
    inner.run.side_effect = [
        _result(hitl=True, a="x", b="y"),
        _result(hitl=False, a="z", b="z"),
    ]
    with patch("agents.multi_agent.consensus_node.CrossProviderConsensusNode", return_value=inner):
        node = ConsensusNode(primary, secondary, use_openai_judge=False, max_self_corrections=2)
        out = node.run("base")
    assert out.state == ConsensusState.CONSENSUS_OK
    assert out.attempts == 2
    assert len(out.self_correction_trace) == 1
    assert inner.run.call_count == 2
    second_prompt = inner.run.call_args_list[1][0][0]
    assert "SELF-CORRECTION" in second_prompt


def test_consensus_node_discrepancy_after_exhausting_retries():
    primary = MagicMock()
    secondary = MagicMock()
    inner = MagicMock()
    inner.run.return_value = _result(hitl=True, a="x", b="y")
    with patch("agents.multi_agent.consensus_node.CrossProviderConsensusNode", return_value=inner):
        node = ConsensusNode(primary, secondary, use_openai_judge=False, max_self_corrections=1)
        out = node.run("base")
    assert out.state == ConsensusState.DISCREPANCY_DETECTED
    assert out.final_answer == ""
    assert inner.run.call_count == 2


def test_consensus_node_both_failed():
    primary = MagicMock()
    secondary = MagicMock()
    inner = MagicMock()
    inner.run.return_value = CrossProviderConsensusResult(
        task_id="t",
        prompt_excerpt="",
        answers=(
            ProviderAnswer("p1", "m", "", error="e1"),
            ProviderAnswer("p2", "m", "", error="e2"),
        ),
        agreement_score=0.0,
        models_agree=False,
        final_text="",
        strategy="both_failed",
        hitl_required=True,
        hitl_reason="both down",
    )
    with patch("agents.multi_agent.consensus_node.CrossProviderConsensusNode", return_value=inner):
        node = ConsensusNode(primary, secondary, use_openai_judge=False, max_self_corrections=0)
        out = node.run("q")
    assert out.state == ConsensusState.BOTH_FAILED
