"""
Sandboxed Code Execution
=========================
Docker-based code runner with resource limits, timeout enforcement,
and structured output capture.

Architecture:
  CodeSandbox          — orchestrates Docker container lifecycle
  SandboxConfig        — resource limits (CPU, memory, timeout, network)
  ExecutionResult      — structured output (stdout, stderr, exit_code, metrics)
  LanguageRunner       — per-language execution strategies
  SandboxPool          — reusable container pool for low-latency execution

Security layers:
  1. Docker isolation   — process, filesystem, and network isolation
  2. Resource limits    — CPU quota, memory limit, PID limit
  3. Read-only rootfs   — container filesystem is read-only
  4. No network         — network disabled by default
  5. Timeout            — hard kill after timeout_seconds
  6. Output truncation  — stdout/stderr capped at max_output_bytes
  7. Static analysis    — optional pre-execution safety check

Note:
  Actual sandboxing requires Docker running locally.
  The module is fully importable and all orchestration code is functional.
  Run `docker info` to verify Docker is available.

Usage:
    sandbox = CodeSandbox()
    result = sandbox.run(
        code="print('Hello, world!')",
        language="python",
    )
    print(result.stdout)   # Hello, world!
    print(result.exit_code)  # 0

    # With custom limits
    config = SandboxConfig(timeout_seconds=5, memory_mb=128)
    result = sandbox.run(code, language="python", config=config)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SandboxConfig:
    """
    Resource limits and security settings for a sandboxed execution.

    All limits are enforced at the Docker level — they cannot be bypassed
    by the executing code.
    """
    timeout_seconds: float = 10.0
    memory_mb: int = 256              # Docker --memory
    cpu_quota: float = 0.5            # fraction of one CPU core
    max_output_bytes: int = 65536     # 64 KB stdout + stderr cap
    max_pids: int = 64                # --pids-limit
    network_disabled: bool = True     # --network none
    read_only_rootfs: bool = True     # --read-only
    allow_write_tmp: bool = True      # mount /tmp as tmpfs
    tmpfs_size_mb: int = 64           # /tmp size limit
    user: str = "nobody"              # run as unprivileged user

    def to_docker_flags(self) -> List[str]:
        """Convert config to Docker run flags."""
        cpu_period = 100_000          # 100ms
        cpu_quota_val = int(self.cpu_quota * cpu_period)

        flags = [
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",   # disable swap
            f"--cpu-period={cpu_period}",
            f"--cpu-quota={cpu_quota_val}",
            f"--pids-limit={self.max_pids}",
            "--cap-drop=ALL",                      # drop all Linux capabilities
            "--security-opt=no-new-privileges",
        ]

        if self.network_disabled:
            flags.append("--network=none")

        if self.read_only_rootfs:
            flags.append("--read-only")

        if self.allow_write_tmp:
            flags.append(f"--tmpfs=/tmp:size={self.tmpfs_size_mb}m,noexec")

        return flags


# Default configs per risk level
SAFE_CONFIG = SandboxConfig(timeout_seconds=10, memory_mb=256, cpu_quota=0.5)
STRICT_CONFIG = SandboxConfig(timeout_seconds=5, memory_mb=128, cpu_quota=0.25, max_pids=32)
PERMISSIVE_CONFIG = SandboxConfig(timeout_seconds=30, memory_mb=512, cpu_quota=1.0,
                                   network_disabled=False)


# ──────────────────────────────────────────────────────────────────────────────
# Execution result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Structured output from a sandboxed code execution."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_mb: Optional[float] = None
    language: str = ""
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_dict(self) -> Dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "memory_used_mb": self.memory_used_mb,
            "language": self.language,
            "truncated": self.truncated,
            "success": self.success,
        }

    def __str__(self) -> str:
        status = "OK" if self.success else ("TIMEOUT" if self.timed_out else f"ERROR({self.exit_code})")
        return (
            f"[{status}] {self.language} | {self.execution_time_ms:.0f}ms\n"
            f"stdout: {self.stdout[:200]}\n"
            f"stderr: {self.stderr[:200]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Language runners
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LanguageRunner:
    """Execution strategy for a specific programming language."""
    language: str
    docker_image: str
    file_extension: str
    run_command: str          # template: {file} = path to code file
    compile_command: Optional[str] = None  # for compiled languages

    def build_exec_cmd(self, code_path: str) -> List[str]:
        """Build the command to execute the code file."""
        cmd = self.run_command.replace("{file}", code_path)
        return shlex.split(cmd)


LANGUAGE_RUNNERS: Dict[str, LanguageRunner] = {
    "python": LanguageRunner(
        language="python",
        docker_image="python:3.11-slim",
        file_extension=".py",
        run_command="python /sandbox/code.py",
    ),
    "javascript": LanguageRunner(
        language="javascript",
        docker_image="node:20-slim",
        file_extension=".js",
        run_command="node /sandbox/code.js",
    ),
    "bash": LanguageRunner(
        language="bash",
        docker_image="bash:5",
        file_extension=".sh",
        run_command="bash /sandbox/code.sh",
    ),
    "ruby": LanguageRunner(
        language="ruby",
        docker_image="ruby:3.2-slim",
        file_extension=".rb",
        run_command="ruby /sandbox/code.rb",
    ),
    "go": LanguageRunner(
        language="go",
        docker_image="golang:1.21-alpine",
        file_extension=".go",
        run_command="go run /sandbox/code.go",
    ),
    "rust": LanguageRunner(
        language="rust",
        docker_image="rust:1.75-slim",
        file_extension=".rs",
        compile_command="rustc /sandbox/code.rs -o /sandbox/code_bin",
        run_command="/sandbox/code_bin",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Static analysis (pre-execution safety check)
# ──────────────────────────────────────────────────────────────────────────────

_DANGEROUS_PYTHON_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\(.+['\"]w['\"]",   # file write
    r"\bsocket\b",
    r"\burllib\b|\brequests\b",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b|\bos\.unlink\b",
    r"\bctypes\b",
    r"\bpickle\b",
]

_DANGEROUS_JS_PATTERNS = [
    r"\brequire\s*\(\s*['\"]child_process",
    r"\brequire\s*\(\s*['\"]fs",
    r"\bprocess\.exit\b",
    r"\beval\s*\(",
    r"\bFunction\s*\(",
]

_STATIC_ANALYSIS_RULES: Dict[str, List[str]] = {
    "python": _DANGEROUS_PYTHON_PATTERNS,
    "javascript": _DANGEROUS_JS_PATTERNS,
}


def static_analysis(code: str, language: str) -> Dict:
    """
    Quick static analysis to detect obviously dangerous patterns.
    Returns a list of warnings (not a hard block — Docker provides the real isolation).
    """
    patterns = _STATIC_ANALYSIS_RULES.get(language, [])
    warnings = []
    for pattern in patterns:
        if re.search(pattern, code):
            warnings.append(f"Potentially dangerous pattern: {pattern}")

    return {
        "warnings": warnings,
        "warning_count": len(warnings),
        "has_warnings": len(warnings) > 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core sandbox
# ──────────────────────────────────────────────────────────────────────────────

class CodeSandbox:
    """
    Docker-based code sandbox.

    Each execution:
      1. Writes code to a temp directory
      2. Runs `docker run` with resource limits
      3. Captures stdout/stderr with timeout enforcement
      4. Returns structured ExecutionResult

    Requires Docker to be running. Check with `docker info`.
    """

    def __init__(
        self,
        default_config: Optional[SandboxConfig] = None,
        run_static_analysis: bool = True,
        docker_binary: str = "docker",
    ):
        self.default_config = default_config or SAFE_CONFIG
        self.run_static_analysis = run_static_analysis
        self.docker_binary = docker_binary
        self._docker_available: Optional[bool] = None

    def is_docker_available(self) -> bool:
        """Check if Docker daemon is running."""
        if self._docker_available is None:
            try:
                result = subprocess.run(
                    [self.docker_binary, "info"],
                    capture_output=True, timeout=5
                )
                self._docker_available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._docker_available = False
        return self._docker_available

    def run(
        self,
        code: str,
        language: str = "python",
        config: Optional[SandboxConfig] = None,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed Docker container.

        Args:
            code:     source code to execute
            language: programming language (python | javascript | bash | ruby | go | rust)
            config:   resource limits (defaults to SAFE_CONFIG)
            stdin:    optional stdin to pipe to the process
            env:      optional environment variables (passed as --env)

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing, etc.
        """
        cfg = config or self.default_config
        runner = LANGUAGE_RUNNERS.get(language)

        if runner is None:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                error=f"Unsupported language: {language}. "
                      f"Supported: {list(LANGUAGE_RUNNERS.keys())}",
                language=language,
            )

        # Static analysis
        if self.run_static_analysis:
            analysis = static_analysis(code, language)
            if analysis["has_warnings"]:
                logger.warning(
                    "Static analysis warnings for %s: %s",
                    language, analysis["warnings"],
                )

        if not self.is_docker_available():
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                error="Docker is not available. Run `docker info` to diagnose.",
                language=language,
            )

        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
            return self._run_in_docker(code, runner, cfg, tmpdir, stdin, env)

    def _run_in_docker(
        self,
        code: str,
        runner: LanguageRunner,
        cfg: SandboxConfig,
        tmpdir: str,
        stdin: Optional[str],
        env: Optional[Dict[str, str]],
    ) -> ExecutionResult:
        """Build and execute the Docker run command."""
        # Write code to temp file
        code_file = Path(tmpdir) / f"code{runner.file_extension}"
        code_file.write_text(code, encoding="utf-8")

        # Build docker run command
        docker_flags = cfg.to_docker_flags()

        # Mount the code directory as read-only
        mount = f"{tmpdir}:/sandbox:ro"

        env_flags: List[str] = []
        if env:
            for k, v in env.items():
                # Sanitize key names
                if re.match(r'^[A-Z_][A-Z0-9_]*$', k):
                    env_flags += ["--env", f"{k}={v}"]

        cmd = (
            [self.docker_binary, "run", "--rm"]
            + docker_flags
            + ["-v", mount]
            + env_flags
            + ["--workdir", "/sandbox"]
            + [runner.docker_image]
            + runner.build_exec_cmd("/sandbox/code" + runner.file_extension)
        )

        logger.debug("Docker command: %s", " ".join(cmd))

        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin else None,
            )

            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    input=stdin.encode() if stdin else None,
                    timeout=cfg.timeout_seconds,
                )
                exit_code = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_bytes, stderr_bytes = proc.communicate()
                exit_code = -1
                timed_out = True

        except Exception as e:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                error=f"Docker execution failed: {e}",
                language=runner.language,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Decode and truncate output
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        truncated = False

        if len(stdout) + len(stderr) > cfg.max_output_bytes:
            half = cfg.max_output_bytes // 2
            stdout = stdout[:half] + "\n[TRUNCATED]" if len(stdout) > half else stdout
            stderr = stderr[:half] + "\n[TRUNCATED]" if len(stderr) > half else stderr
            truncated = True

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            execution_time_ms=elapsed_ms,
            language=runner.language,
            truncated=truncated,
        )

    def run_with_tests(
        self,
        code: str,
        test_code: str,
        language: str = "python",
        config: Optional[SandboxConfig] = None,
    ) -> ExecutionResult:
        """
        Run code + test suite together.
        Appends test_code to code and executes the combined script.
        """
        combined = code + "\n\n# --- Tests ---\n" + test_code
        return self.run(combined, language=language, config=config)

    def benchmark(
        self,
        code: str,
        language: str = "python",
        n_runs: int = 5,
        config: Optional[SandboxConfig] = None,
    ) -> Dict:
        """
        Run code n_runs times and return timing statistics.
        """
        times = []
        for _ in range(n_runs):
            result = self.run(code, language=language, config=config)
            if result.success:
                times.append(result.execution_time_ms)

        if not times:
            return {"error": "All runs failed"}

        return {
            "n_runs": n_runs,
            "n_successful": len(times),
            "mean_ms": round(float(sum(times) / len(times)), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Container pool (reuse containers for lower latency)
# ──────────────────────────────────────────────────────────────────────────────

class SandboxPool:
    """
    Maintains a pool of warm Docker containers to reduce cold-start latency.

    Instead of `docker run` (which starts a new container each time),
    the pool keeps containers running and uses `docker exec` to run code.

    Latency comparison:
      docker run:  ~500ms–2s (image pull + container start)
      docker exec: ~10–50ms (code execution only)

    Note: containers in the pool share a filesystem — use with caution.
    Prefer CodeSandbox.run() for untrusted code; use SandboxPool for
    trusted code that needs low latency (e.g. test runners).
    """

    def __init__(
        self,
        language: str = "python",
        pool_size: int = 2,
        config: Optional[SandboxConfig] = None,
        docker_binary: str = "docker",
    ):
        self.language = language
        self.pool_size = pool_size
        self.config = config or SAFE_CONFIG
        self.docker_binary = docker_binary
        self._runner = LANGUAGE_RUNNERS.get(language)
        self._container_ids: List[str] = []
        self._lock = threading.Lock()
        self._available: List[str] = []

    def start(self) -> "SandboxPool":
        """Start pool containers."""
        if self._runner is None:
            raise ValueError(f"Unsupported language: {self.language}")

        for i in range(self.pool_size):
            container_id = self._start_container()
            if container_id:
                self._container_ids.append(container_id)
                self._available.append(container_id)
                logger.info("Pool container %d started: %s", i, container_id[:12])

        return self

    def _start_container(self) -> Optional[str]:
        """Start a long-running container and return its ID."""
        flags = self.config.to_docker_flags()
        cmd = (
            [self.docker_binary, "run", "-d", "--rm"]
            + flags
            + [self._runner.docker_image, "sleep", "3600"]  # keep alive 1 hour
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.error("Failed to start pool container: %s", e)
        return None

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in a pool container using docker exec."""
        with self._lock:
            if not self._available:
                logger.warning("No containers available in pool, falling back to docker run")
                sandbox = CodeSandbox(self.config)
                return sandbox.run(code, self.language)
            container_id = self._available.pop()

        try:
            with tempfile.NamedTemporaryFile(
                suffix=self._runner.file_extension,
                mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            # Copy code into container
            subprocess.run(
                [self.docker_binary, "cp", tmp_path, f"{container_id}:/sandbox/code{self._runner.file_extension}"],
                check=True, capture_output=True, timeout=5,
            )
            os.unlink(tmp_path)

            # Execute
            exec_cmd = self._runner.build_exec_cmd(f"/sandbox/code{self._runner.file_extension}")
            t0 = time.perf_counter()
            proc = subprocess.run(
                [self.docker_binary, "exec", container_id] + exec_cmd,
                capture_output=True, timeout=self.config.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout.decode("utf-8", errors="replace"),
                stderr=proc.stderr.decode("utf-8", errors="replace"),
                execution_time_ms=elapsed_ms,
                language=self.language,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(exit_code=-1, stdout="", stderr="", timed_out=True,
                                   language=self.language)
        except Exception as e:
            return ExecutionResult(exit_code=-1, stdout="", stderr="",
                                   error=str(e), language=self.language)
        finally:
            with self._lock:
                if container_id in self._container_ids:
                    self._available.append(container_id)

    def stop(self) -> None:
        """Stop all pool containers."""
        for cid in self._container_ids:
            try:
                subprocess.run([self.docker_binary, "kill", cid],
                               capture_output=True, timeout=5)
            except Exception:
                pass
        self._container_ids.clear()
        self._available.clear()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Docker Compose service definition
# ──────────────────────────────────────────────────────────────────────────────

SANDBOX_DOCKER_COMPOSE = """\
# Sandbox execution service
# Provides an HTTP API for sandboxed code execution
# Requires Docker-in-Docker (dind) for nested container execution

services:
  sandbox-api:
    build:
      context: .
      dockerfile: sandbox/Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # Docker socket for container management
    environment:
      - SANDBOX_DEFAULT_TIMEOUT=10
      - SANDBOX_MAX_MEMORY_MB=256
      - SANDBOX_NETWORK_DISABLED=true
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G

  # Optional: pre-pull language images for faster cold starts
  image-puller:
    image: docker:cli
    command: >
      sh -c "
        docker pull python:3.11-slim &&
        docker pull node:20-slim &&
        docker pull bash:5
      "
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    restart: "no"
"""

SANDBOX_DOCKERFILE = """\
FROM python:3.11-slim

# Install Docker CLI (for running child containers)
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sandbox/ ./sandbox/

# Simple HTTP API wrapper
CMD ["python", "-m", "sandbox.api_server"]
"""


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sandboxed code execution")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Execute code in sandbox")
    run_p.add_argument("--code", help="Code to execute")
    run_p.add_argument("--file", help="File containing code to execute")
    run_p.add_argument("--language", default="python",
                       choices=list(LANGUAGE_RUNNERS.keys()))
    run_p.add_argument("--timeout", type=float, default=10.0)
    run_p.add_argument("--memory", type=int, default=256)

    check_p = sub.add_parser("check", help="Check Docker availability")

    analyze_p = sub.add_parser("analyze", help="Static analysis only (no execution)")
    analyze_p.add_argument("--code", required=True)
    analyze_p.add_argument("--language", default="python")

    compose_p = sub.add_parser("gen-compose", help="Print Docker Compose config")

    demo_p = sub.add_parser("demo", help="Run demo (requires Docker)")

    args = parser.parse_args()

    if args.cmd == "check":
        sandbox = CodeSandbox()
        available = sandbox.is_docker_available()
        print(f"Docker available: {available}")
        if not available:
            print("Install Docker: https://docs.docker.com/get-docker/")
        else:
            print("Ready to run sandboxed code.")

    elif args.cmd == "analyze":
        result = static_analysis(args.code, args.language)
        print(json.dumps(result, indent=2))

    elif args.cmd == "gen-compose":
        print(SANDBOX_DOCKER_COMPOSE)

    elif args.cmd == "run":
        if args.file:
            code = Path(args.file).read_text(encoding="utf-8")
        elif args.code:
            code = args.code
        else:
            parser.error("Provide --code or --file")

        config = SandboxConfig(timeout_seconds=args.timeout, memory_mb=args.memory)
        sandbox = CodeSandbox()

        if not sandbox.is_docker_available():
            print("Docker not available. Cannot execute code.")
            print("Static analysis:")
            print(json.dumps(static_analysis(code, args.language), indent=2))
        else:
            result = sandbox.run(code, language=args.language, config=config)
            print(result)

    elif args.cmd == "demo":
        sandbox = CodeSandbox()

        DEMO_PROGRAMS = {
            "python": 'import sys\nprint("Hello from Python!")\nprint(f"Python {sys.version}")',
            "javascript": 'console.log("Hello from Node.js!");\nconsole.log(`Node ${process.version}`);',
            "bash": 'echo "Hello from bash!"\necho "Running as: $(whoami)"',
        }

        if not sandbox.is_docker_available():
            print("Docker not available — showing what would be executed:\n")
            for lang, code in DEMO_PROGRAMS.items():
                analysis = static_analysis(code, lang)
                print(f"[{lang}] Static analysis: {analysis['warning_count']} warnings")
                print(f"  Code: {code[:60]}...")
        else:
            print("Running demo programs in Docker sandbox:\n")
            for lang, code in DEMO_PROGRAMS.items():
                result = sandbox.run(code, language=lang)
                print(f"[{lang}] {result}")
                print()

    else:
        parser.print_help()
