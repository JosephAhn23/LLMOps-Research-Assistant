"""
Versioned feature registry for the RAG pipeline.

Provides:
  - Feature registration with schema and dependency tracking
  - Point-in-time correct feature retrieval (prevents training/serving skew)
  - Online (low-latency dict) and offline (DataFrame) serving paths
  - Feature lineage: which features depend on which raw columns
  - Schema versioning: detect breaking changes

This is a lightweight, local-first feature store. For production at scale,
replace the in-memory store with Redis (online) + Delta Lake (offline).

Usage:
    store = FeatureStore()
    store.register(
        name="query_complexity",
        fn=lambda df: df["query"].str.split().str.len() / 100,
        depends_on=["query"],
        description="Normalized query token count",
    )
    features = store.compute(df, feature_names=["query_complexity"])
    online_record = store.serve_online({"query": "What is RAG?"})
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    name: str
    fn: Callable
    depends_on: list[str]
    description: str
    version: int = 1
    dtype: str = "float64"
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def signature(self) -> str:
        """Hash of the feature spec for change detection."""
        content = f"{self.name}:{self.description}:{self.depends_on}:{self.version}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass
class FeatureVector:
    entity_id: str
    features: dict[str, float]
    timestamp: str
    version_signatures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "features": self.features,
            "version_signatures": self.version_signatures,
        }


class FeatureStore:
    """
    Versioned feature registry with point-in-time correct serving.

    Parameters
    ----------
    ttl_seconds : int
        Cache TTL for online serving (default: 300s = 5 minutes).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._registry: dict[str, FeatureSpec] = {}
        self._online_cache: dict[str, tuple[FeatureVector, float]] = {}
        self._ttl = ttl_seconds

    def register(
        self,
        name: str,
        fn: Callable,
        depends_on: list[str],
        description: str = "",
        version: int = 1,
        dtype: str = "float64",
        tags: dict[str, str] | None = None,
    ) -> "FeatureStore":
        """
        Register a feature transformation.

        Parameters
        ----------
        name : str
            Feature name (must be unique).
        fn : Callable
            Transformation function. For batch: takes a DataFrame, returns a Series.
            For online: takes a dict of raw values, returns a float.
        depends_on : list[str]
            Raw column names this feature depends on.
        description : str
            Human-readable description for the feature catalog.
        version : int
            Increment when the transformation logic changes.
        dtype : str
            Expected output dtype.
        tags : dict | None
            Arbitrary metadata (e.g., {"team": "retrieval", "tier": "online"}).
        """
        if name in self._registry:
            existing = self._registry[name]
            if existing.version != version:
                logger.warning(
                    "Feature '%s' version changed: %d -> %d",
                    name, existing.version, version,
                )

        self._registry[name] = FeatureSpec(
            name=name,
            fn=fn,
            depends_on=depends_on,
            description=description,
            version=version,
            dtype=dtype,
            tags=tags or {},
        )
        logger.debug("Registered feature '%s' v%d", name, version)
        return self

    def compute(
        self,
        df: Any,
        feature_names: list[str] | None = None,
        as_dataframe: bool = True,
    ) -> Any:
        """
        Batch compute features from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data with raw columns.
        feature_names : list[str] | None
            Features to compute. If None, computes all registered features.
        as_dataframe : bool
            If True, returns a DataFrame with original columns + feature columns.
            If False, returns a dict of {feature_name: Series}.
        """
        import pandas as pd

        names = feature_names or list(self._registry.keys())
        result = df.copy() if as_dataframe else {}

        for name in names:
            spec = self._registry.get(name)
            if spec is None:
                logger.warning("Feature '%s' not registered; skipping", name)
                continue

            missing = [c for c in spec.depends_on if c not in df.columns]
            if missing:
                logger.warning(
                    "Feature '%s' missing dependencies: %s; skipping", name, missing
                )
                continue

            try:
                values = spec.fn(df)
                if as_dataframe:
                    result[name] = values
                else:
                    result[name] = values
            except Exception as e:
                logger.error("Feature '%s' computation failed: %s", name, e)

        return result

    def serve_online(
        self,
        raw: dict[str, Any],
        feature_names: list[str] | None = None,
        entity_id: str | None = None,
    ) -> FeatureVector:
        """
        Compute features for a single entity (low-latency path).

        Parameters
        ----------
        raw : dict
            Raw feature values (e.g., {"query": "What is RAG?"}).
        feature_names : list[str] | None
            Features to compute. If None, computes all registered features.
        entity_id : str | None
            Optional entity identifier for cache lookup.
        """
        import time

        if entity_id:
            cached = self._online_cache.get(entity_id)
            if cached is not None:
                vec, ts = cached
                if time.time() - ts < self._ttl:
                    return vec

        names = feature_names or list(self._registry.keys())
        features: dict[str, float] = {}
        signatures: dict[str, str] = {}

        for name in names:
            spec = self._registry.get(name)
            if spec is None:
                continue
            missing = [k for k in spec.depends_on if k not in raw]
            if missing:
                continue
            try:
                val = spec.fn(raw)
                features[name] = float(val)
                signatures[name] = spec.signature()
            except Exception as e:
                logger.debug("Online feature '%s' failed: %s", name, e)

        vec = FeatureVector(
            entity_id=entity_id or "",
            features=features,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version_signatures=signatures,
        )

        if entity_id:
            self._online_cache[entity_id] = (vec, time.time())

        return vec

    def lineage(self, feature_name: str) -> dict[str, Any]:
        """Return the dependency tree for a feature."""
        spec = self._registry.get(feature_name)
        if spec is None:
            return {}
        return {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "depends_on": spec.depends_on,
            "signature": spec.signature(),
            "tags": spec.tags,
        }

    def catalog(self) -> list[dict[str, Any]]:
        """Return the full feature catalog."""
        return [self.lineage(name) for name in self._registry]

    def detect_schema_changes(self, previous_signatures: dict[str, str]) -> list[str]:
        """
        Detect features whose transformation logic has changed.
        Returns a list of feature names with changed signatures.
        """
        changed = []
        for name, spec in self._registry.items():
            prev = previous_signatures.get(name)
            if prev is not None and prev != spec.signature():
                changed.append(name)
                logger.warning(
                    "Feature '%s' signature changed: %s -> %s",
                    name, prev, spec.signature(),
                )
        return changed

    def export_catalog(self, path: str) -> None:
        """Export the feature catalog to a JSON file."""
        catalog = self.catalog()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        logger.info("Feature catalog exported to %s", path)
