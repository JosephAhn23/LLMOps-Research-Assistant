"""Tests for sandbox AutonomousDebugLoop (mocked execution)."""
from __future__ import annotations

from unittest.mock import MagicMock

from sandbox.code_sandbox import ExecutionResult
from sandbox.debug_loop import AutonomousDebugLoop, default_artifact_ok


def test_default_artifact_ok_csv_and_marker():
    ok = ExecutionResult(
        exit_code=0,
        stdout="col_a,col_b,col_c\n1,2,3\nARTIFACT_OK\n",
        stderr="",
        language="python",
    )
    assert default_artifact_ok(ok) is True

    bad = ExecutionResult(exit_code=0, stdout="just text", stderr="", language="python")
    assert default_artifact_ok(bad) is False


def test_debug_loop_revise_until_success():
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        ExecutionResult(exit_code=1, stdout="", stderr="TypeError: bad", language="python"),
        ExecutionResult(
            exit_code=0,
            stdout="a,b\n1,2\nARTIFACT_OK",
            stderr="",
            language="python",
        ),
    ]

    def revise(_code: str, _fb: str) -> str:
        return "fixed code"

    loop = AutonomousDebugLoop(sandbox=sandbox, max_iterations=3)
    result = loop.run_until_artifact("broken", revise)
    assert result.success is True
    assert result.iterations == 2
    assert sandbox.run.call_count == 2


def test_debug_loop_gives_up():
    sandbox = MagicMock()
    sandbox.run.return_value = ExecutionResult(
        exit_code=1, stdout="", stderr="SyntaxError", language="python"
    )

    def revise(_code: str, _fb: str) -> str:
        return "still broken"

    loop = AutonomousDebugLoop(sandbox=sandbox, max_iterations=2)
    result = loop.run_until_artifact("x", revise)
    assert result.success is False
    assert result.iterations == 2
