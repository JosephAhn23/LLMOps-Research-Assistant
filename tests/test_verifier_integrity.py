"""Scientific integrity / reward-hacking heuristics on VerifierAgent."""
from agents.multi_agent.base_agent import AgentTask
from agents.multi_agent.verifier_agent import VerifierAgent, scan_scientific_integrity


def test_scan_detects_hand_waving_and_plot():
    text = (
        "Using matplotlib we plot the curve. Clearly the optimum is unique without any derivation."
    )
    r = scan_scientific_integrity(text)
    assert r["scientific_context"] is True
    assert "hand_waving" in " ".join(r["integrity_flags"])


def test_scan_deep_reason_on_multiple_flags():
    text = (
        "numpy array analysis. random.seed(0)\nplt.plot([1, 2, 3, 4, 5])\n"
        "Clearly convergence is obvious."
    )
    r = scan_scientific_integrity(text)
    assert r["deep_reason_recommended"] is True


def test_verifier_metadata_includes_integrity():
    v = VerifierAgent()
    task = AgentTask(
        query="q",
        context={
            "research_result": (
                "The theorem follows from the plot. matplotlib figure 1 shows fit. "
                "Obviously no further proof is required."
            ),
            "retrieved_context": [
                {"text": "matplotlib figure 1 illustrates model fit for the dataset.", "score": 0.9},
            ],
        },
    )
    out = v.process(task)
    assert "scientific_integrity" in out.metadata
    assert isinstance(out.metadata["scientific_integrity"]["integrity_flags"], list)
