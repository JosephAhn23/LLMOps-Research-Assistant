"""Self-healing pytest loop (Docker path mocked)."""

from unittest.mock import MagicMock, patch

from sandbox.code_sandbox import ExecutionResult
from sandbox.self_healing_loop import SelfHealingLoop


def test_self_healing_stops_on_green() -> None:
    loop = SelfHealingLoop()
    good = ExecutionResult(exit_code=0, stdout="1 passed", stderr="", language="bash")

    with patch.object(loop, "_docker_pytest", return_value=good):
        def revise(files, fb):
            raise AssertionError("no revise on green")

        r = loop.run_until_pytest_green(
            {"t.py": "def test_x(): assert 1"},
            revise,
            pytest_target="t.py",
            max_rounds=3,
        )
    assert r.success is True
    assert r.rounds == 1


def test_self_healing_calls_revise_on_red() -> None:
    loop = SelfHealingLoop()
    bad = ExecutionResult(exit_code=1, stdout="", stderr="AssertionError", language="bash")
    good = ExecutionResult(exit_code=0, stdout="ok", stderr="", language="bash")

    with patch.object(loop, "_docker_pytest", side_effect=[bad, good]):
        calls: list[str] = []

        def revise(files, fb):
            calls.append(fb[:30])
            return {**files, "t.py": "def test_x(): assert True\n"}

        r = loop.run_until_pytest_green({"t.py": "def test_x(): assert False"}, revise, max_rounds=4)
    assert r.success is True
    assert len(calls) == 1
