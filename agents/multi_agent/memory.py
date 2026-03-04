"""
Shared memory store for the multi-agent system.

Architecture:
  - ShortTermMemory: in-process dict with TTL eviction (per-session context)
  - LongTermMemory: vector store integration (FAISS) for semantic retrieval
  - WorkingMemory: structured task state shared across agents in a pipeline run

Key design decisions:
  - Version vectors for optimistic concurrency (no locks needed for reads)
  - TTL eviction prevents unbounded memory growth in long-running sessions
  - Semantic search on long-term memory enables agents to recall relevant prior work
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    key: str
    value: Any
    author: str
    version: int
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def content_hash(self) -> str:
        return hashlib.sha256(str(self.value).encode()).hexdigest()[:12]


class ShortTermMemory:
    """
    In-process key-value store with TTL eviction and optimistic locking.

    Used for: sharing intermediate results between agents within a single
    pipeline run (research output, critique, verified claims).
    """

    def __init__(self, default_ttl_seconds: float = 3600.0):
        self._store: Dict[str, MemoryEntry] = {}
        self._default_ttl = default_ttl_seconds
        self._write_log: List[Dict] = []

    def write(
        self,
        key: str,
        value: Any,
        author: str,
        ttl_seconds: Optional[float] = None,
        expected_version: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Write with optimistic locking.
        Raises ValueError on version conflict (caller should re-read and retry).
        """
        existing = self._store.get(key)
        current_version = existing.version if existing else 0

        if expected_version is not None and expected_version != current_version:
            raise ValueError(
                f"Version conflict on '{key}': expected v{expected_version}, "
                f"got v{current_version}. Re-read and retry."
            )

        new_version = current_version + 1
        entry = MemoryEntry(
            key=key,
            value=value,
            author=author,
            version=new_version,
            ttl_seconds=ttl_seconds or self._default_ttl,
            tags=tags or [],
        )
        self._store[key] = entry
        self._write_log.append({
            "key": key,
            "author": author,
            "version": new_version,
            "hash": entry.content_hash(),
            "ts": time.time(),
        })
        logger.debug("Memory write: key=%s author=%s v%d", key, author, new_version)
        return new_version

    def read(self, key: str) -> Tuple[Any, int]:
        """Returns (value, version). Returns (None, 0) if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            return None, 0
        if entry.is_expired():
            del self._store[key]
            return None, 0
        return entry.value, entry.version

    def read_all(self, tag: Optional[str] = None) -> Dict[str, Any]:
        """Read all non-expired entries, optionally filtered by tag."""
        result = {}
        expired_keys = []
        for key, entry in self._store.items():
            if entry.is_expired():
                expired_keys.append(key)
                continue
            if tag and tag not in entry.tags:
                continue
            result[key] = entry.value
        for k in expired_keys:
            del self._store[k]
        return result

    def evict_expired(self) -> int:
        expired = [k for k, e in self._store.items() if e.is_expired()]
        for k in expired:
            del self._store[k]
        return len(expired)

    def audit_trail(self, key: Optional[str] = None) -> List[Dict]:
        if key:
            return [e for e in self._write_log if e["key"] == key]
        return list(self._write_log)

    def snapshot(self) -> Dict[str, Any]:
        """Full snapshot of non-expired state for debugging."""
        return {k: v.value for k, v in self._store.items() if not v.is_expired()}


class LongTermMemory:
    """
    Vector store-backed long-term memory for semantic retrieval.

    Agents can store important findings (e.g., verified facts, past answers)
    and retrieve semantically similar entries for future tasks.

    In production: backed by FAISS index or pgvector.
    Here: cosine similarity over numpy arrays.
    """

    def __init__(self, embedding_dim: int = 64):
        self._entries: List[Dict[str, Any]] = []
        self._embeddings: List[Any] = []
        self._dim = embedding_dim

    def store(self, key: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Store a text entry with a simulated embedding."""
        embedding = self._embed(text)
        self._entries.append({"key": key, "text": text, "metadata": metadata or {}})
        self._embeddings.append(embedding)
        logger.debug("LTM store: key=%s len=%d", key, len(self._entries))

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search: returns top_k most similar entries."""
        if not self._entries:
            return []

        try:
            import numpy as np
            q_emb = self._embed(query)
            emb_matrix = np.array(self._embeddings)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            normed = emb_matrix / (norms + 1e-9)
            scores = normed @ q_norm
            top_idx = scores.argsort()[::-1][:top_k]
            return [
                {**self._entries[i], "score": float(scores[i])}
                for i in top_idx
            ]
        except ImportError:
            return self._entries[:top_k]

    def _embed(self, text: str):
        """Deterministic pseudo-embedding from text (no model needed)."""
        try:
            import numpy as np
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed)
            return rng.randn(self._dim).astype(np.float32)
        except ImportError:
            return [0.0] * self._dim

    def __len__(self) -> int:
        return len(self._entries)


class WorkingMemory:
    """
    Structured working memory for a single pipeline run.
    Combines short-term (fast, keyed) and long-term (semantic) stores.
    """

    def __init__(self, session_id: str, ttl_seconds: float = 1800.0):
        self.session_id = session_id
        self.short_term = ShortTermMemory(default_ttl_seconds=ttl_seconds)
        self.long_term = LongTermMemory()

    def set(self, key: str, value: Any, author: str, **kwargs) -> int:
        return self.short_term.write(key, value, author, **kwargs)

    def get(self, key: str) -> Tuple[Any, int]:
        return self.short_term.read(key)

    def remember(self, key: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Persist important findings to long-term memory."""
        self.long_term.store(key, text, metadata)

    def recall(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic retrieval from long-term memory."""
        return self.long_term.retrieve(query, top_k)

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "short_term_keys": list(self.short_term.snapshot().keys()),
            "long_term_entries": len(self.long_term),
        }
