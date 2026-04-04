"""
Strict execution orchestrator — no user-visible solution until Docker run + tests pass.

Combines agent code with an auto-generated or supplied test block, runs inside
:class:`CodeSandbox`, captures a structured **execution log**, and loops on
stack traces via an injectable ``revise`` callable (typically an LLM).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from sandbox.code_sandbox import CodeSandbox, ExecutionResult, SandboxConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLogEntry:
    round_index: int
    timestamp_iso: str
    exit_code: int
    success: bool
    stdout_excerpt: str
    stderr_excerpt: str
    timed_out: bool

    def format_line(self) -> str:
        return (
            f"[round {self.round_index}] exit={self.exit_code} ok={self.success} "
            f"timeout={self.timed_out} | stderr={self.stderr_excerpt[:200]!r}"
        )


@dataclass
class StrictExecutionResult:
    success: bool
    final_code: str
    execution_log_text: str
    entries: List[ExecutionLogEntry] = field(default_factory=list)
    last_result: Optional[ExecutionResult] = None


class StrictExecutionOrchestrator:
    """
    Run ``implementation_code`` + ``test_code`` in an isolated container; require
    green execution before returning. The **execution log** is intended to ship
    with the final answer as proof of execution.
    """

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        *,
        max_rounds: int = 5,
        language: str = "python",
        config: Optional[SandboxConfig] = None,
    ) -> None:
        self.sandbox = sandbox or CodeSandbox()
        self.max_rounds = max_rounds
        self.language = language
        self.config = config

    def run_until_green(
        self,
        implementation_code: str,
        test_code: str,
        revise: Callable[[str, str], str],
    ) -> StrictExecutionResult:
        code = implementation_code.strip()
        entries: List[ExecutionLogEntry] = []
        last: Optional[ExecutionResult] = None

        for r in range(1, self.max_rounds + 1):
            last = self.sandbox.run_with_tests(code, test_code, language=self.language, config=self.config)
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ent = ExecutionLogEntry(
                round_index=r,
                timestamp_iso=ts,
                exit_code=last.exit_code,
                success=last.success,
                stdout_excerpt=(last.stdout or "")[:4000],
                stderr_excerpt=(last.stderr or "")[:4000],
                timed_out=last.timed_out,
            )
            entries.append(ent)
            logger.info("StrictExecution: %s", ent.format_line())

            if last.success:
                log_text = self._format_log(entries)
                return StrictExecutionResult(
                    success=True,
                    final_code=code,
                    execution_log_text=log_text,
                    entries=entries,
                    last_result=last,
                )

            feedback = f"STDOUT:\n{last.stdout}\n\nSTDERR:\n{last.stderr}\n"
            if last.error:
                feedback += f"\nSANDBOX_ERROR: {last.error}\n"
            code = revise(code, feedback).strip()
            if not code:
                break

        return StrictExecutionResult(
            success=False,
            final_code=code,
            execution_log_text=self._format_log(entries),
            entries=entries,
            last_result=last,
        )

    @staticmethod
    def _format_log(entries: List[ExecutionLogEntry]) -> str:
        lines = ["=== EXECUTION LOG (Docker sandbox) ==="]
        for e in entries:
            lines.append(e.format_line())
            if e.stdout_excerpt.strip():
                lines.append(f"  stdout: {e.stdout_excerpt[:500]}…" if len(e.stdout_excerpt) > 500 else f"  stdout: {e.stdout_excerpt}")
        lines.append("=== END EXECUTION LOG ===")
        return "\n".join(lines)
