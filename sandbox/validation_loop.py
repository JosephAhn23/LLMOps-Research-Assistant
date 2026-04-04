"""
High-fidelity validation: execute proposed code and compare output to the agent's claim.

Runs up to ``max_rounds`` (default 3) self-correction cycles: on mismatch or sandbox
failure, the traceback and stdout are formatted for a reviser (typically an LLM)
before the user sees a "verified" result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from sandbox.code_sandbox import CodeSandbox, ExecutionResult, SandboxConfig
from sandbox.debug_loop import format_traceback_for_agent

logger = logging.getLogger(__name__)

_FLOAT = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _FLOAT.finditer(text):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def default_claim_matcher(claim: str, stdout: str, stderr: str = "") -> bool:
    """
    Default verification: if both sides have floats, require pairwise approximate
    equality (within rel_tol); else require substantial stdout overlap with claim.
    """
    combined_out = (stdout + "\n" + stderr).strip()
    if not combined_out:
        return False
    cn = extract_numbers(claim)
    sn = extract_numbers(combined_out)
    if cn and sn:
        if len(sn) < len(cn):
            return False
        for a, b in zip(cn, sn[: len(cn)]):
            if a == 0:
                if abs(b) > 1e-9:
                    return False
            else:
                if abs(a - b) > max(1e-6, abs(a) * 1e-4):
                    return False
        return True
    claim_l = claim.lower().strip()
    out_l = combined_out.lower()
    if len(claim_l) < 8:
        return claim_l in out_l
    # token overlap heuristic
    wc = set(re.findall(r"[a-zA-Z]{4,}", claim_l))
    wo = set(re.findall(r"[a-zA-Z]{4,}", out_l))
    if not wc:
        return True
    inter = len(wc & wo)
    return inter / max(len(wc), 1) >= 0.4


@dataclass
class ValidationLoopResult:
    claim_verified: bool
    final_code: str
    last_execution: ExecutionResult
    rounds_used: int
    matcher_notes: str = ""
    feedback_trace: List[str] = field(default_factory=list)


class ValidationLoop:
    """
    Execute code in Docker sandbox; compare to ``claim`` with ``matcher``.

    ``revise(previous_code, feedback) -> new_code`` is invoked until verification
    succeeds or ``max_rounds`` is exhausted.
    """

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        *,
        max_rounds: int = 3,
        language: str = "python",
        config: Optional[SandboxConfig] = None,
    ) -> None:
        self.sandbox = sandbox or CodeSandbox()
        self.max_rounds = max_rounds
        self.language = language
        self.config = config

    def run(
        self,
        claim: str,
        initial_code: str,
        revise: Callable[[str, str], str],
        matcher: Optional[Callable[[str, str, str], bool]] = None,
    ) -> ValidationLoopResult:
        match_fn = matcher or (lambda c, o, e: default_claim_matcher(c, o, e))
        code = initial_code.strip()
        trace: List[str] = []
        last = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="",
            error="not executed",
            language=self.language,
        )
        notes = ""

        for round_i in range(1, self.max_rounds + 1):
            last = self.sandbox.run(code, language=self.language, config=self.config)
            ok_exec = last.success
            stdout = last.stdout or ""
            stderr = last.stderr or ""
            verified = ok_exec and match_fn(claim, stdout, stderr)
            notes = f"exec_ok={ok_exec} verified={verified}"

            if verified:
                logger.info("ValidationLoop: claim verified in round %d", round_i)
                return ValidationLoopResult(
                    claim_verified=True,
                    final_code=code,
                    last_execution=last,
                    rounds_used=round_i,
                    matcher_notes=notes,
                    feedback_trace=trace,
                )

            fb_parts = [
                "The code output does not match the stated claim.",
                f"CLAIM:\n{claim.strip()[:4000]}",
                "",
                format_traceback_for_agent(code, last),
                "",
                "Fix the code so execution succeeds and the printed/computed result supports the claim.",
            ]
            feedback = "\n".join(fb_parts)
            trace.append(feedback[:6000])
            code = revise(code, feedback).strip()
            if not code:
                break

        return ValidationLoopResult(
            claim_verified=False,
            final_code=code,
            last_execution=last,
            rounds_used=self.max_rounds,
            matcher_notes=notes,
            feedback_trace=trace,
        )
