"""
Self-healing TDD loop — pytest must pass inside an isolated Docker workspace before “done”.

The agent supplies a bundle of files (implementation + tests), we mount them in a
temporary directory, run ``python -m pytest`` in ``python:3.11-slim``, and feed
stdout/stderr back to a ``revise`` callable until green or ``max_rounds``.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sandbox.code_sandbox import ExecutionResult, SandboxConfig, SAFE_CONFIG

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "python:3.11-slim"


@dataclass
class SelfHealingLoopResult:
    success: bool
    files: Dict[str, str]
    rounds: int
    execution_log: str
    last_result: Optional[ExecutionResult] = None


class SelfHealingLoop:
    """
    Run pytest in Docker on a writable temp workspace; iterate with ``revise``.
    """

    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        image: str = _DEFAULT_IMAGE,
        config: Optional[SandboxConfig] = None,
        pip_install: str = "pytest",
    ) -> None:
        self.docker_binary = docker_binary
        self.image = image
        self.config = config or SAFE_CONFIG
        self.pip_install = pip_install

    def run_until_pytest_green(
        self,
        files: Dict[str, str],
        revise: Callable[[Dict[str, str], str], Dict[str, str]],
        *,
        pytest_target: str = ".",
        max_rounds: int = 8,
        extra_pip: Optional[str] = None,
    ) -> SelfHealingLoopResult:
        """
        Args:
            files: relative path → file contents (e.g. ``{"solution.py": "...", "test_solution.py": "..."}``).
            revise: ``(files, feedback) -> updated_files``; feedback is stdout+stderr from pytest.
            pytest_target: path relative to workspace root passed to pytest.
        """
        log_lines: List[str] = ["=== SELF-HEALING PYTEST LOG (Docker) ==="]
        current = {k: v for k, v in files.items()}
        last: Optional[ExecutionResult] = None

        for rnd in range(1, max_rounds + 1):
            with tempfile.TemporaryDirectory(prefix="selfheal_") as tmp:
                root = Path(tmp)
                for rel, content in current.items():
                    p = root / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")

                last = self._docker_pytest(root, pytest_target, extra_pip=extra_pip)
                log_lines.append(
                    f"[round {rnd}] exit={last.exit_code} ok={last.success} "
                    f"timeout={last.timed_out}"
                )
                if last.stderr:
                    log_lines.append(f"  stderr: {last.stderr[:1500]}")
                if last.stdout:
                    log_lines.append(f"  stdout: {last.stdout[:1500]}")

                if last.success:
                    log_lines.append("=== END LOG — GREEN ===")
                    return SelfHealingLoopResult(
                        success=True,
                        files=current,
                        rounds=rnd,
                        execution_log="\n".join(log_lines),
                        last_result=last,
                    )

                feedback = f"STDOUT:\n{last.stdout}\n\nSTDERR:\n{last.stderr}\n"
                if last.error:
                    feedback += f"\nSANDBOX: {last.error}\n"
                current = revise(current, feedback)
                if not current:
                    break

        log_lines.append("=== END LOG — FAILED ===")
        return SelfHealingLoopResult(
            success=False,
            files=current,
            rounds=max_rounds,
            execution_log="\n".join(log_lines),
            last_result=last,
        )

    def _docker_pytest(
        self,
        workspace: Path,
        pytest_target: str,
        *,
        extra_pip: Optional[str] = None,
    ) -> ExecutionResult:
        cfg = self.config
        flags = cfg.to_docker_flags()
        mount = f"{workspace.resolve()}:/workspace:rw"
        pip_pkgs = self.pip_install
        if extra_pip:
            pip_pkgs = f"{pip_pkgs} {extra_pip}"
        inner = (
            f"pip install -q {pip_pkgs} && "
            f"python -m pytest {pytest_target} -q --tb=short"
        )
        cmd = (
            [self.docker_binary, "run", "--rm"]
            + flags
            + ["-v", mount, "-w", "/workspace", self.image, "bash", "-lc", inner]
        )
        logger.debug("SelfHealing docker: %s", " ".join(cmd))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_seconds,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                exit_code=proc.returncode or 0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
                execution_time_ms=elapsed,
                language="bash",
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
                error="pytest docker timeout",
                execution_time_ms=elapsed,
                language="bash",
            )
        except FileNotFoundError:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                error=f"Docker not found: {self.docker_binary}",
                language="bash",
            )
        except Exception as exc:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(exc),
                language="bash",
            )
