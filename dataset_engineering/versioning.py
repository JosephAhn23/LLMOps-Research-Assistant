"""
Dataset versioning with DVC-style content-addressed storage.

Provides immutable, reproducible dataset snapshots with:
  - SHA-256 content hashing for deduplication
  - Lineage tracking (parent -> child transformations)
  - Metadata snapshots (schema, row count, column stats)
  - DVC integration when available; falls back to local manifest files

Usage:
    registry = DatasetRegistry(store_dir=".dataset_store")
    v1 = registry.register("train_rag", df, tags={"source": "synthetic"})
    print(v1.version_id)

    v2 = registry.register("train_rag", df_cleaned, parent=v1.version_id,
                            tags={"transform": "dedup+quality_filter"})

    lineage = registry.lineage("train_rag")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    name: str
    version_id: str
    parent_id: str | None
    created_at: str
    row_count: int
    column_names: list[str]
    content_hash: str
    tags: dict[str, str]
    stats: dict[str, Any] = field(default_factory=dict)
    dvc_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "stats": self.stats,
            "dvc_path": self.dvc_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetVersion":
        return cls(**d)


def _hash_dataframe(df: Any) -> str:
    """SHA-256 hash of a DataFrame's content (order-independent)."""
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            # Sort for determinism, hash the CSV bytes
            content = df.sort_values(by=list(df.columns)).to_csv(index=False).encode()
            return hashlib.sha256(content).hexdigest()[:16]
    except ImportError:
        pass
    # Fallback: hash the repr
    return hashlib.sha256(repr(df).encode()).hexdigest()[:16]


def _compute_stats(df: Any) -> dict[str, Any]:
    """Compute basic column statistics for a DataFrame."""
    stats: dict[str, Any] = {}
    try:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return stats
        for col in df.columns:
            col_stats: dict[str, Any] = {"dtype": str(df[col].dtype)}
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                col_stats.update({
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "null_rate": float(df[col].isna().mean()),
                })
            else:
                col_stats.update({
                    "null_rate": float(df[col].isna().mean()),
                    "unique_count": int(df[col].nunique()),
                })
            stats[col] = col_stats
    except Exception as e:
        logger.debug("Stats computation failed: %s", e)
    return stats


class DatasetRegistry:
    """
    Immutable dataset version registry with lineage tracking.

    Parameters
    ----------
    store_dir : str
        Directory for storing version manifests.
    use_dvc : bool
        If True, attempt to use DVC for data storage. Falls back to local
        manifest-only mode if DVC is not configured.
    """

    def __init__(
        self,
        store_dir: str = ".dataset_store",
        use_dvc: bool = False,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.use_dvc = use_dvc
        self._manifest_path = self.store_dir / "manifest.json"
        self._versions: dict[str, list[DatasetVersion]] = self._load_manifest()

    def _load_manifest(self) -> dict[str, list[DatasetVersion]]:
        if self._manifest_path.exists():
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return {
                name: [DatasetVersion.from_dict(v) for v in versions]
                for name, versions in raw.items()
            }
        return {}

    def _save_manifest(self) -> None:
        raw = {
            name: [v.to_dict() for v in versions]
            for name, versions in self._versions.items()
        }
        self._manifest_path.write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )

    def register(
        self,
        name: str,
        df: Any,
        parent: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> DatasetVersion:
        """
        Register a new dataset version.

        Parameters
        ----------
        name : str
            Dataset name (e.g., "train_rag", "eval_set").
        df : DataFrame
            The dataset to version.
        parent : str | None
            Version ID of the parent dataset (for lineage tracking).
        tags : dict | None
            Arbitrary metadata (e.g., {"transform": "dedup", "source": "synthetic"}).
        """
        content_hash = _hash_dataframe(df)
        version_id = f"{name}-{content_hash}"

        # Check for duplicate content
        existing = self._versions.get(name, [])
        for v in existing:
            if v.content_hash == content_hash:
                logger.info("Dataset '%s' already registered as %s", name, v.version_id)
                return v

        try:
            row_count = len(df)
            column_names = list(df.columns) if hasattr(df, "columns") else []
        except Exception:
            row_count = 0
            column_names = []

        version = DatasetVersion(
            name=name,
            version_id=version_id,
            parent_id=parent,
            created_at=datetime.now(timezone.utc).isoformat(),
            row_count=row_count,
            column_names=column_names,
            content_hash=content_hash,
            tags=tags or {},
            stats=_compute_stats(df),
        )

        if name not in self._versions:
            self._versions[name] = []
        self._versions[name].append(version)
        self._save_manifest()

        logger.info(
            "Registered dataset '%s' version %s (%d rows)",
            name, version_id, row_count,
        )
        return version

    def get(self, name: str, version_id: str | None = None) -> DatasetVersion | None:
        """Get a specific version (latest if version_id is None)."""
        versions = self._versions.get(name, [])
        if not versions:
            return None
        if version_id is None:
            return versions[-1]
        return next((v for v in versions if v.version_id == version_id), None)

    def lineage(self, name: str) -> list[DatasetVersion]:
        """Return all versions of a dataset in chronological order."""
        return self._versions.get(name, [])

    def lineage_tree(self, name: str) -> str:
        """Render the lineage as an ASCII tree."""
        versions = self.lineage(name)
        if not versions:
            return f"No versions found for '{name}'"

        lines = [f"Dataset: {name}"]
        for v in versions:
            parent_str = f" <- {v.parent_id}" if v.parent_id else " (root)"
            tag_str = ", ".join(f"{k}={val}" for k, val in v.tags.items())
            lines.append(
                f"  {v.version_id}  {v.row_count:,} rows  "
                f"{v.created_at[:10]}{parent_str}"
                + (f"  [{tag_str}]" if tag_str else "")
            )
        return "\n".join(lines)

    def diff(self, name: str, v1_id: str, v2_id: str) -> dict[str, Any]:
        """Compare two versions of a dataset."""
        v1 = self.get(name, v1_id)
        v2 = self.get(name, v2_id)
        if v1 is None or v2 is None:
            raise ValueError(f"Version not found: {v1_id} or {v2_id}")

        added_cols = set(v2.column_names) - set(v1.column_names)
        removed_cols = set(v1.column_names) - set(v2.column_names)

        return {
            "row_count_delta": v2.row_count - v1.row_count,
            "added_columns": sorted(added_cols),
            "removed_columns": sorted(removed_cols),
            "content_changed": v1.content_hash != v2.content_hash,
            "v1_tags": v1.tags,
            "v2_tags": v2.tags,
        }
