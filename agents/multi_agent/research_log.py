"""
Durable research session log — "lessons learned" across long agent runs.

Append-only JSONL per ``session_id`` under ``RESEARCH_LOG_DIR`` (default
``outputs/research_logs``). Inject into prompts via :meth:`format_for_prompt`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(os.getenv("RESEARCH_LOG_DIR", "outputs/research_logs"))


class ResearchLog:
    """Thread-safe append + read for one logical research session."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base = Path(base_dir) if base_dir is not None else _DEFAULT_DIR
        self._locks: Dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:128]
        if not safe:
            safe = "default"
        return self._base / f"{safe}.jsonl"

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._meta:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def append_lesson(
        self,
        session_id: str,
        lesson: str,
        *,
        kind: str = "lesson",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a lesson (e.g. sign error, wrong equation)."""
        lesson = lesson.strip()
        if not lesson:
            return
        self._base.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "kind": kind,
            "lesson": lesson[:8000],
            "metadata": metadata or {},
        }
        path = self._path(session_id)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock_for(session_id):
            path.open("a", encoding="utf-8").write(line)
        logger.info("ResearchLog: appended %s to %s", kind, path.name)

    def read_lessons(self, session_id: str, *, max_entries: int = 50) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        with self._lock_for(session_id):
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        out: List[Dict[str, Any]] = []
        for line in lines[-max_entries:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def format_for_prompt(self, session_id: str, *, max_entries: int = 20) -> str:
        """Compact block suitable for prepending to a user or system message."""
        rows = self.read_lessons(session_id, max_entries=max_entries)
        if not rows:
            return ""
        lines = ["[Prior lessons from this research session — do not repeat these mistakes:]"]
        for r in rows:
            lesson = r.get("lesson", "")
            if lesson:
                lines.append(f"- {lesson}")
        return "\n".join(lines) + "\n"
