"""
dataset_engineering/feature_store.py
--------------------------------------
Lightweight feature store for LLMOps pipelines.

Features:
  - Register feature definitions with schema + transformation logic
  - Versioned feature snapshots (point-in-time correct)
  - Online (dict lookup) + offline (parquet) materialization
  - Feature lineage and dependency graph
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    name: str
    description: str
    dtype: str                       # "float", "int", "str", "embedding"
    transform_fn: Callable           # raw_df -> Series or ndarray
    dependencies: list[str] = field(default_factory=list)
    version: str = "v1.0"
    tags: list[str] = field(default_factory=list)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return self.transform_fn(df)


@dataclass
class FeatureSnapshot:
    feature_name: str
    version: str
    data: pd.Series
    computed_at: str
    content_hash: str

    @classmethod
    def create(cls, name: str, version: str, data: pd.Series) -> "FeatureSnapshot":
        from datetime import datetime, timezone
        h = hashlib.sha256(data.to_json().encode()).hexdigest()
        return cls(
            feature_name=name,
            version=version,
            data=data,
            computed_at=datetime.now(timezone.utc).isoformat(),
            content_hash=h,
        )


# Legacy aliases for backwards compatibility
FeatureSpec = FeatureDefinition
FeatureVector = FeatureSnapshot


class FeatureStore:
    """
    Manages feature definitions, computation, and materialization.

    Usage
    -----
    >>> store = FeatureStore(root_dir="features/")
    >>> store.register(FeatureDefinition(
    ...     name="query_len",
    ...     description="Character length of query",
    ...     dtype="int",
    ...     transform_fn=lambda df: df["query"].str.len(),
    ... ))
    >>> features = store.compute_all(df)
    >>> store.materialize(features, split="train")
    """

    def __init__(self, root_dir: str | Path = "features"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, FeatureDefinition] = {}
        self._snapshots: dict[str, FeatureSnapshot] = {}

    def register(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        if feature.name in self._registry:
            logger.info("Overwriting feature: %s", feature.name)
        self._registry[feature.name] = feature
        logger.info("Registered feature: %s (%s)", feature.name, feature.dtype)

    def register_many(self, features: list[FeatureDefinition]) -> None:
        for f in features:
            self.register(f)

    def compute(self, name: str, df: pd.DataFrame) -> FeatureSnapshot:
        """Compute a single feature and cache the snapshot."""
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' not registered")
        feat = self._registry[name]
        data = feat.compute(df)
        snapshot = FeatureSnapshot.create(name, feat.version, data)
        self._snapshots[name] = snapshot
        logger.info("Computed feature '%s': shape=%s", name, data.shape)
        return snapshot

    def compute_all(
        self, df: pd.DataFrame, tags: list[str] | None = None
    ) -> dict[str, FeatureSnapshot]:
        """Compute all registered features (optionally filtered by tag)."""
        to_compute = [
            f for f in self._registry.values()
            if tags is None or any(t in f.tags for t in tags)
        ]
        snapshots = {}
        for feat in self._resolve_dependency_order(to_compute):
            snap = self.compute(feat.name, df)
            snapshots[feat.name] = snap
        return snapshots

    def to_dataframe(self, snapshots: dict[str, FeatureSnapshot]) -> pd.DataFrame:
        """Assemble snapshots into a feature DataFrame."""
        return pd.DataFrame({name: snap.data for name, snap in snapshots.items()})

    def materialize(
        self,
        snapshots: dict[str, FeatureSnapshot],
        split: str = "train",
        fmt: str = "parquet",
    ) -> Path:
        """Write feature matrix to disk."""
        df = self.to_dataframe(snapshots)
        out_path = self.root_dir / f"{split}_features.{fmt}"

        if fmt == "parquet":
            df.to_parquet(out_path, index=False)
        elif fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        meta = {
            "split": split,
            "features": [
                {"name": name, "version": snap.version, "hash": snap.content_hash}
                for name, snap in snapshots.items()
            ],
        }
        (self.root_dir / f"{split}_features.meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        logger.info("Materialized %d features -> %s", len(df.columns), out_path)
        return out_path

    def load(self, split: str = "train", fmt: str = "parquet") -> pd.DataFrame:
        """Load a materialized feature matrix."""
        path = self.root_dir / f"{split}_features.{fmt}"
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def lineage_graph(self) -> dict[str, list[str]]:
        """Return the feature dependency graph as an adjacency dict."""
        return {f.name: f.dependencies for f in self._registry.values()}

    def catalog(self) -> pd.DataFrame:
        """Return a DataFrame describing all registered features."""
        return pd.DataFrame([
            {
                "name": f.name,
                "description": f.description,
                "dtype": f.dtype,
                "version": f.version,
                "tags": ", ".join(f.tags),
                "dependencies": ", ".join(f.dependencies),
            }
            for f in self._registry.values()
        ])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_dependency_order(
        self, features: list[FeatureDefinition]
    ) -> list[FeatureDefinition]:
        """Topological sort so dependencies are computed before dependents."""
        name_to_feat = {f.name: f for f in features}
        order: list[FeatureDefinition] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            feat = name_to_feat.get(name)
            if feat is None:
                return
            for dep in feat.dependencies:
                visit(dep)
            order.append(feat)

        for f in features:
            visit(f.name)
        return order
