"""Strict execution orchestrator with mocked sandbox."""

from unittest.mock import MagicMock

from sandbox.code_sandbox import ExecutionResult
from sandbox.strict_execution_orchestrator import StrictExecutionOrchestrator


def test_orchestrator_stops_on_first_success() -> None:
    sandbox = MagicMock()
    sandbox.run_with_tests.return_value = ExecutionResult(
        exit_code=0,
        stdout="ok",
        stderr="",
        language="python",
    )
    orch = StrictExecutionOrchestrator(sandbox=sandbox, max_rounds=5)

    def revise(code: str, feedback: str) -> str:
        raise AssertionError("revise should not run on success")

    r = orch.run_until_green("x = 1", "assert x == 1", revise)
    assert r.success is True
    assert "EXECUTION LOG" in r.execution_log_text
    assert sandbox.run_with_tests.call_count == 1


def test_orchestrator_revises_on_failure_then_succeeds() -> None:
    sandbox = MagicMock()
    sandbox.run_with_tests.side_effect = [
        ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            language="python",
        ),
        ExecutionResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            language="python",
        ),
    ]
    orch = StrictExecutionOrchestrator(sandbox=sandbox, max_rounds=5)
    calls: list[str] = []

    def revise(code: str, feedback: str) -> str:
        calls.append(feedback)
        return code + "\n# fixed"

    r = orch.run_until_green("bad", "assert False", revise)
    assert r.success is True
    assert len(calls) == 1
    assert "AssertionError" in calls[0]
    assert sandbox.run_with_tests.call_count == 2


def test_orchestrator_fails_after_max_rounds() -> None:
    sandbox = MagicMock()
    sandbox.run_with_tests.return_value = ExecutionResult(
        exit_code=1,
        stdout="",
        stderr="fail",
        language="python",
    )
    orch = StrictExecutionOrchestrator(sandbox=sandbox, max_rounds=2)

    def revise(code: str, feedback: str) -> str:
        return code

    r = orch.run_until_green("c", "assert 0", revise)
    assert r.success is False
    assert len(r.entries) == 2
