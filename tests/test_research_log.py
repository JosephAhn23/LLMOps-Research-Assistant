from agents.multi_agent.research_log import ResearchLog


def test_append_and_format(tmp_path):
    log = ResearchLog(base_dir=tmp_path)
    log.append_lesson("sess-a", "Equation 4 had a sign error; use minus in front of the kinetic term.")
    log.append_lesson("sess-a", "Do not trust the smooth plot without raw residuals.")
    block = log.format_for_prompt("sess-a")
    assert "sign error" in block
    assert "smooth plot" in block
