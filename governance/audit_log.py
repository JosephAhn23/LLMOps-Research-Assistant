"""
Cryptographic audit log with SHA-256 hash chain.

Every entry is linked to the previous entry via its hash, creating an
immutable chain. Tampering with any entry invalidates all subsequent hashes.

This is the same principle used in blockchain, but much simpler:
we don't need consensus or distributed validation — just immutability
for regulatory compliance (GDPR, SOC2, HIPAA audit requirements).

Entries cover:
  - Dataset creation and lineage
  - Model training runs
  - Stage promotions (staging -> production)
  - Deployments
  - Inference requests (with PII-redacted prompts)
  - Human review decisions (HITL)
  - Fairness evaluation results
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    entry_id: str
    event_type: str
    actor: str
    payload: Dict[str, Any]
    timestamp: str
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "entry_id": self.entry_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CryptoAuditLog:
    """
    Append-only audit log with SHA-256 hash chain.

    Each entry's hash depends on:
    1. Its own content (event_type, actor, payload, timestamp)
    2. The previous entry's hash (chain linkage)

    Verification: recompute all hashes and check chain integrity.
    Any tampering breaks the chain at the tampered entry.

    Persistence: optionally writes to JSONL file for durability.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, persist_path: Optional[str] = None):
        self._entries: List[AuditEntry] = []
        self._persist_path = persist_path
        if persist_path:
            self._load(persist_path)

    def log(
        self,
        event_type: str,
        actor: str,
        payload: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> AuditEntry:
        """Append a new entry to the chain."""
        prev_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        entry_id = hashlib.sha256(
            f"{event_type}{actor}{ts}{prev_hash}".encode()
        ).hexdigest()[:16]

        entry = AuditEntry(
            entry_id=entry_id,
            event_type=event_type,
            actor=actor,
            payload=payload,
            timestamp=ts,
            prev_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._entries.append(entry)

        if self._persist_path:
            self._append_to_file(entry)

        logger.debug(
            "Audit: [%s] %s by %s (hash=%s)",
            event_type, entry_id, actor, entry.entry_hash[:12],
        )
        return entry

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of the entire chain.
        Returns (is_valid, error_message).
        """
        if not self._entries:
            return True, None

        expected_prev = self.GENESIS_HASH
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != expected_prev:
                return False, (
                    f"Chain broken at entry {i} (id={entry.entry_id}): "
                    f"expected prev_hash={expected_prev[:12]}..., "
                    f"got {entry.prev_hash[:12]}..."
                )
            recomputed = entry.compute_hash()
            if recomputed != entry.entry_hash:
                return False, (
                    f"Hash mismatch at entry {i} (id={entry.entry_id}): "
                    f"stored={entry.entry_hash[:12]}..., "
                    f"computed={recomputed[:12]}..."
                )
            expected_prev = entry.entry_hash

        return True, None

    def get_entries(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[AuditEntry]:
        results = self._entries
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if actor:
            results = [e for e in results if e.actor == actor]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results

    def get_model_lineage(self, model_name: str) -> List[AuditEntry]:
        return [
            e for e in self._entries
            if e.payload.get("model_name") == model_name
        ]

    def tail(self, n: int = 10) -> List[AuditEntry]:
        return self._entries[-n:]

    def export_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return len(self._entries)

    def _append_to_file(self, entry: AuditEntry) -> None:
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to persist audit entry: %s", e)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self._entries.append(AuditEntry(**d))
            logger.info("Loaded %d audit entries from %s.", len(self._entries), path)
        except Exception as e:
            logger.error("Failed to load audit log from %s: %s", path, e)

    def __len__(self) -> int:
        return len(self._entries)

    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
