from governance.economic_judge import EconomicJudge


def test_within_caps_no_hitl():
    j = EconomicJudge(max_iterations=5, max_usd_per_task=2.0)
    v = j.evaluate(
        estimated_input_tokens=10_000,
        estimated_output_tokens=2_000,
        estimated_iterations=2,
        task_label="small fix",
        value_note="Unblocks prod.",
    )
    assert v.requires_hitl is False
    assert v.estimated_usd < 2.0


def test_excess_iterations_triggers_hitl():
    j = EconomicJudge(max_iterations=5, max_usd_per_task=100.0)
    v = j.evaluate(
        estimated_input_tokens=1_000,
        estimated_output_tokens=500,
        estimated_iterations=10,
        task_label="CSS tweak",
    )
    assert v.requires_hitl is True
    assert "iteration" in v.rationale.lower()


def test_excess_usd_triggers_hitl():
    j = EconomicJudge(max_iterations=20, max_usd_per_task=1.0, price_per_1k_input=1.0)
    v = j.evaluate(
        estimated_input_tokens=5_000,
        estimated_output_tokens=0,
        estimated_iterations=1,
        task_label="expensive run",
    )
    assert v.requires_hitl is True
    assert "usd" in v.rationale.lower() or "$" in v.roi_prompt
