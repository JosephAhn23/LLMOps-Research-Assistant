"""
Autonomous debug loop: execute generated code, feed tracebacks back, repeat.

Use when a research agent emits Python (or other supported sandbox languages):
the loop does not return success until execution succeeds **and** a minimal
artifact contract is satisfied (stdout heuristic or explicit marker).

Requires Docker for :class:`sandbox.code_sandbox.CodeSandbox` by default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from sandbox.code_sandbox import CodeSandbox, ExecutionResult, SandboxConfig

logger = logging.getLogger(__name__)


def default_artifact_ok(result: ExecutionResult) -> bool:
    """
    Success if process exited 0 and stdout looks like a deliberate artifact.

    Heuristics:
      * Line ``ARTIFACT_OK`` (explicit contract), or
      * First line looks CSV-like (>= 2 commas), or
      * ``PLOT_OK`` for matplotlib-style scripts that print confirmation.
    """
    if not result.success:
        return False
    out = result.stdout.strip()
    if not out:
        return False
    if "ARTIFACT_OK" in out or "PLOT_OK" in out:
        return True
    first = out.splitlines()[0]
    if first.count(",") >= 2 and len(first) >= 8:
        return True
    return False


def format_traceback_for_agent(previous_code: str, result: ExecutionResult) -> str:
    """Build a user message for the next model turn."""
    parts = [
        "Your code was executed in a sandbox and did not satisfy the artifact contract.",
        "",
        "```python",
        previous_code.strip(),
        "```",
        "",
        f"exit_code={result.exit_code} timed_out={result.timed_out}",
        "",
        "stderr:",
        result.stderr.strip() or "(empty)",
        "",
        "stdout:",
        result.stdout.strip()[:8000] or "(empty)",
    ]
    if result.error:
        parts.extend(["", f"sandbox_error: {result.error}"])
    parts.extend(
        [
            "",
            "Rewrite the full script. On success print ARTIFACT_OK after the data output.",
        ]
    )
    return "\n".join(parts)


@dataclass
class DebugLoopResult:
    success: bool
    final_code: str
    last_execution: ExecutionResult
    iterations: int
    feedback_trace: List[str] = field(default_factory=list)


class AutonomousDebugLoop:
    """
    Run ``sandbox.run`` in a loop, calling ``revise`` with traceback context until
    ``artifact_ok`` passes or ``max_iterations`` is exceeded.

    ``revise(previous_code, feedback_message) -> new_code`` is typically implemented
    by calling an LLM; in tests, use a stub.
    """

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        *,
        max_iterations: int = 5,
        language: str = "python",
        config: Optional[SandboxConfig] = None,
    ) -> None:
        self.sandbox = sandbox or CodeSandbox()
        self.max_iterations = max_iterations
        self.language = language
        self.config = config

    def run_until_artifact(
        self,
        initial_code: str,
        revise: Callable[[str, str], str],
        artifact_ok: Callable[[ExecutionResult], bool] = default_artifact_ok,
    ) -> DebugLoopResult:
        code = initial_code.strip()
        trace: List[str] = []
        last = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="",
            error="not executed",
            language=self.language,
        )

        for i in range(1, self.max_iterations + 1):
            last = self.sandbox.run(code, language=self.language, config=self.config)
            if artifact_ok(last):
                logger.info("AutonomousDebugLoop: success after %d iteration(s)", i)
                return DebugLoopResult(True, code, last, i, trace)

            feedback = format_traceback_for_agent(code, last)
            trace.append(feedback[:4000])
            code = revise(code, feedback).strip()
            if not code:
                logger.warning("AutonomousDebugLoop: reviser returned empty code")
                break

        return DebugLoopResult(False, code, last, self.max_iterations, trace)
