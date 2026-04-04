"""
Repository symbol map for LLM context (static, AST-based).

Builds a compact outline of public-ish classes and functions so prompts can
reference real symbols without hallucinating paths. Optional ``pyright`` JSON
pass can be added later; this module works with stdlib only.

Usage:
    from context_engineering.symbol_map import generate_symbol_map_text
    text = generate_symbol_map_text(Path("."), max_lines=400)
    system += "\\n\\n## Repository symbol map\\n" + text
"""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


def generate_symbol_map_text(
    repo_root: Path,
    *,
    package_roots: Optional[Iterable[str]] = None,
    max_files: int = 120,
    max_lines: int = 500,
) -> str:
    """
    Walk ``package_roots`` (top-level dirs under repo) and emit ``module: Class / def``.
    """
    root = Path(repo_root).resolve()
    roots = list(package_roots) if package_roots else _default_package_roots(root)
    lines: List[str] = []
    seen: Set[Path] = set()
    n_files = 0
    for pr in roots:
        base = root / pr
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if n_files >= max_files:
                break
            if not path.is_file() or path in seen:
                continue
            if _should_skip(path, root):
                continue
            seen.add(path)
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(root)
            for row in _outline_file(rel.as_posix(), src):
                lines.append(row)
            n_files += 1
        if n_files >= max_files:
            break

    if not lines:
        return "(no symbols collected — check package_roots / paths)"
    header = f"# Symbol map ({n_files} files, cap {max_files})\n"
    body = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        body += f"\n... truncated ({len(lines) - max_lines} more lines)"
    return header + body


def _default_package_roots(root: Path) -> List[str]:
    candidates = [
        "agents",
        "api",
        "ingestion",
        "mlops",
        "governance",
        "sandbox",
        "context_engineering",
        "eval",
        "safety",
        "multi_agent",
    ]
    return [c for c in candidates if (root / c).is_dir()]


def _should_skip(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    if set(parts) & _DEFAULT_SKIP_DIRS:
        return True
    if "tests" in parts and path.name.startswith("test_"):
        return True
    return False


def _outline_file(module_path: str, source: str) -> List[str]:
    lines: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"{module_path}: <syntax error>"]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            methods = [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            ][:8]
            ms = f" ({', '.join(methods)})" if methods else ""
            lines.append(f"{module_path}: class {node.name}{ms}")
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            lines.append(f"{module_path}: def {node.name}()")
    if not lines:
        lines.append(f"{module_path}: <no top-level public classes/functions>")
    return lines


def try_pyright_symbol_dump(repo_root: Path, *, timeout_s: float = 60.0) -> Optional[str]:
    """
    If ``pyright`` is on PATH, return JSON diagnostics summary (optional enrichment).
    Returns None if unavailable or failed.
    """
    root = Path(repo_root).resolve()
    try:
        proc = subprocess.run(
            ["pyright", str(root), "--outputjson"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.info("pyright not used for symbol map: %s", exc)
        return None
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return None
    # Keep raw JSON truncated for prompt injection
    out = proc.stdout.strip()
    return out[:8000] + ("..." if len(out) > 8000 else "")


def build_prompt_injection_block(repo_root: Path, *, use_pyright: bool = False) -> str:
    ast_block = generate_symbol_map_text(repo_root)
    if not use_pyright:
        return ast_block
    pj = try_pyright_symbol_dump(repo_root)
    if pj:
        return ast_block + "\n\n## Pyright (truncated)\n" + pj
    return ast_block
