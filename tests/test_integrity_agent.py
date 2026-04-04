"""Tests for governance.integrity_agent."""
from __future__ import annotations

from governance.integrity_agent import IntegrityAgent


def test_passes_clean_answer_with_citation():
    agent = IntegrityAgent()
    text = "The mass gap follows from the stated boundary conditions [source_1]."
    ok, report = agent.passes_gate(text)
    assert ok
    assert report.blocking() == []


def test_fails_uncited_numeric_claim():
    agent = IntegrityAgent()
    text = "The beam energy is 6.4 GeV and therefore scattering peaks at 30 degrees."
    ok, report = agent.passes_gate(text)
    assert not ok
    assert any(v.rule_id == "uncited_strong_claim" for v in report.blocking())


def test_assumption_phrase_blocks_by_default():
    agent = IntegrityAgent()
    text = "We assume harmonic trapping; the frequency is 100 Hz [source_1]."
    ok, report = agent.passes_gate(text)
    assert not ok
    assert any("assumption" in v.rule_id for v in report.violations)


def test_hedging_warning_not_blocking_unless_configured():
    agent = IntegrityAgent(fail_on_hedging=False)
    text = "It is possible that the noise floor dominates [source_1]."
    ok, report = agent.passes_gate(text)
    assert ok
    assert any(v.rule_id.startswith("hedging") for v in report.violations)


def test_citation_index_out_of_range():
    agent = IntegrityAgent()
    text = "See [source_99] for details."
    ok, report = agent.passes_gate(text, chunks=[{"text": "only one"}])
    assert not ok
    assert any(v.rule_id == "bad_source_index" for v in report.blocking())


def test_citation_substantiated_by_chunk_overlap():
    agent = IntegrityAgent()
    chunks = [{"text": "The boundary conditions fix the mass gap at the edge."}]
    text = "The mass gap follows from boundary conditions [source_1]."
    ok, report = agent.passes_gate(text, chunks=chunks)
    assert ok
