"""
Reproduce-first (TDD) debugging: red → green → full regression.

Workflow for agents:
  1. Write a failing test that reproduces the bug → run until pytest exits non-zero
     (at least one failure). No production fix before this "red" phase succeeds.
  2. Apply the minimal fix → same test file must pass ("green").
  3. Run the full test suite to guard against regressions.

This module only orchestrates subprocess ``pytest``; the agent supplies paths and edits.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class PytestOutcome:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def failed(self) -> bool:
        return self.exit_code != 0


@dataclass
class TDDPhaseResult:
    name: str
    ok: bool
    outcome: PytestOutcome
    message: str = ""


@dataclass
class TDDDebugLoopResult:
    red: TDDPhaseResult
    green: Optional[TDDPhaseResult] = None
    regression: Optional[TDDPhaseResult] = None
    success: bool = False
    errors: List[str] = field(default_factory=list)


def run_pytest(
    targets: Sequence[str],
    cwd: Path,
    *,
    extra_args: Optional[Sequence[str]] = None,
    timeout_s: float = 300.0,
) -> PytestOutcome:
    """Run pytest as a module (same interpreter)."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", *targets]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return PytestOutcome(proc.returncode, proc.stdout or "", proc.stderr or "")


class TDDDebugLoop:
    """
    Enforces reproduce-first debugging using pytest exit codes.

    - ``ensure_red``: pytest must fail (exit != 0) — proves repro test catches the bug.
    - ``ensure_green``: pytest must pass (exit == 0) — repro test passes after fix.
    - ``run_full_regression``: entire suite (default ``tests``) must pass.
    """

    def __init__(self, project_root: Path, *, full_suite_path: str = "tests"):
        self.project_root = Path(project_root).resolve()
        self.full_suite_path = full_suite_path

    def ensure_red(self, test_target: str, *, extra_args: Optional[Sequence[str]] = None) -> TDDPhaseResult:
        """Require at least one failing test (typical pytest exit code 1)."""
        out = run_pytest([test_target], self.project_root, extra_args=extra_args)
        # pytest: 0 = all passed, 1 = failures, 5 = no tests collected
        if out.exit_code == 0:
            ok = False
            msg = "RED phase failed: tests passed — add or fix the reproduction so it fails first."
        elif out.exit_code == 5:
            ok = False
            msg = "RED phase failed: no tests collected."
        elif out.exit_code == 1:
            ok = True
            msg = "RED phase OK: pytest reported test failures (repro active)."
        else:
            ok = False
            msg = f"RED phase inconclusive: pytest exit {out.exit_code}. stderr={out.stderr[:300]!r}"
        logger.info("TDD red phase exit=%s ok=%s", out.exit_code, ok)
        return TDDPhaseResult("red", ok, out, msg)

    def ensure_green(self, test_target: str, *, extra_args: Optional[Sequence[str]] = None) -> TDDPhaseResult:
        """After fix, repro tests must pass."""
        out = run_pytest([test_target], self.project_root, extra_args=extra_args)
        ok = out.exit_code == 0
        msg = "GREEN phase OK." if ok else "GREEN phase failed: tests still failing."
        logger.info("TDD green phase exit=%s ok=%s", out.exit_code, ok)
        return TDDPhaseResult("green", ok, out, msg)

    def run_full_regression(
        self,
        *,
        extra_args: Optional[Sequence[str]] = None,
    ) -> TDDPhaseResult:
        """Run full suite (default ``tests/``)."""
        target = self.full_suite_path
        if not (self.project_root / target).exists():
            out = PytestOutcome(5, "", f"missing {target}")
            return TDDPhaseResult("regression", False, out, f"Path {target!r} not found.")
        out = run_pytest([target], self.project_root, extra_args=extra_args)
        ok = out.exit_code == 0
        msg = "Regression suite OK." if ok else "Regression suite failed."
        return TDDPhaseResult("regression", ok, out, msg)

    def run_cycle_after_fix(
        self,
        repro_test_target: str,
        *,
        extra_args: Optional[Sequence[str]] = None,
        skip_red_check: bool = False,
    ) -> TDDDebugLoopResult:
        """
        Run green + full regression. Call ``ensure_red`` yourself first unless
        ``skip_red_check=True`` (e.g. red already verified in a prior step).
        """
        errors: List[str] = []
        red = TDDPhaseResult("red", True, PytestOutcome(0, "", ""), "skipped")
        if not skip_red_check:
            red = self.ensure_red(repro_test_target, extra_args=extra_args)
            if not red.ok:
                errors.append(red.message)
                return TDDDebugLoopResult(red=red, success=False, errors=errors)

        green = self.ensure_green(repro_test_target, extra_args=extra_args)
        if not green.ok:
            errors.append(green.message)
            return TDDDebugLoopResult(red=red, green=green, success=False, errors=errors)

        reg = self.run_full_regression(extra_args=extra_args)
        if not reg.ok:
            errors.append(reg.message)
            return TDDDebugLoopResult(red=red, green=green, regression=reg, success=False, errors=errors)

        return TDDDebugLoopResult(
            red=red, green=green, regression=reg, success=True, errors=[],
        )
