"""
Feature Store implementation on Delta Lake.

Covers:
  - Feature group registration and versioning
  - Point-in-time correct feature retrieval (no data leakage)
  - Online vs offline store split
  - Feature lineage tracking
  - Training dataset generation with join keys

Architecture:
  Offline store: Delta Lake tables (S3 / DBFS) with time-travel queries
  Online store:  Redis hash (low-latency serving at inference time)
  Registry:      MLflow tracking server (feature metadata + schema)

Usage:
    store = FeatureStore()
    store.register_feature_group("user_query_stats", df, primary_key="user_id")
    training_df = store.get_training_dataset(entity_df, feature_groups=["user_query_stats"])
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Group Schema
# ---------------------------------------------------------------------------

@dataclass
class FeatureSchema:
    name: str
    dtype: str
    nullable: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureGroup:
    name: str
    primary_key: str
    event_time_col: str
    features: List[FeatureSchema]
    description: str = ""
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    owner: str = "llmops"
    tags: Dict[str, str] = field(default_factory=dict)

    def schema_hash(self) -> str:
        schema_repr = json.dumps(
            [asdict(f) for f in self.features], sort_keys=True
        )
        return hashlib.sha256(schema_repr.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_hash"] = self.schema_hash()
        return d


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Delta Lake-backed feature store with online serving via Redis.

    Provides:
    - Feature group registration and schema validation
    - Point-in-time correct joins for training datasets
    - Online feature serving (Redis) for inference
    - Feature lineage: which training runs consumed which feature version
    """

    def __init__(
        self,
        offline_path: str = "delta/feature_store",
        online_backend: str = "memory",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.offline_path = offline_path
        self.online_backend = online_backend
        self._registry: Dict[str, FeatureGroup] = {}
        self._online_store: Dict[str, Dict[str, Any]] = {}
        self._lineage: List[Dict] = []

        try:
            from mlops.compat import mlflow
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            self._mlflow = mlflow
        except ImportError:
            self._mlflow = None

        self._redis = None
        if online_backend == "redis":
            self._redis = self._connect_redis()

    def _connect_redis(self):
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            logger.info("Connected to Redis for online feature serving.")
            return r
        except Exception as e:
            logger.warning("Redis unavailable (%s). Using in-memory online store.", e)
            return None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_feature_group(
        self,
        name: str,
        features: List[FeatureSchema],
        primary_key: str,
        event_time_col: str = "event_time",
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> FeatureGroup:
        """Register a feature group. Increments version if schema changes."""
        existing = self._registry.get(name)
        version = 1
        if existing:
            new_group = FeatureGroup(
                name=name, primary_key=primary_key, event_time_col=event_time_col,
                features=features, description=description, version=existing.version,
                tags=tags or {},
            )
            if new_group.schema_hash() != existing.schema_hash():
                version = existing.version + 1
                logger.info("Schema change detected for '%s' — bumping to v%d.", name, version)
            else:
                version = existing.version

        group = FeatureGroup(
            name=name, primary_key=primary_key, event_time_col=event_time_col,
            features=features, description=description, version=version,
            tags=tags or {},
        )
        self._registry[name] = group

        if self._mlflow:
            with self._mlflow.start_run(run_name=f"feature-group-{name}-v{version}"):
                self._mlflow.log_params({
                    "feature_group": name,
                    "version": version,
                    "primary_key": primary_key,
                    "n_features": len(features),
                    "schema_hash": group.schema_hash(),
                })
                self._mlflow.set_tags(tags or {})

        logger.info("Registered feature group '%s' v%d (%d features).", name, version, len(features))
        return group

    def list_feature_groups(self) -> List[Dict]:
        return [g.to_dict() for g in self._registry.values()]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_features(
        self,
        group_name: str,
        records: List[Dict[str, Any]],
        mode: str = "append",
    ) -> None:
        """
        Write feature records to offline Delta Lake store.
        In production: df.write.format('delta').mode(mode).save(path)
        Here: persists to in-memory store + logs schema to MLflow.
        """
        group = self._registry.get(group_name)
        if not group:
            raise KeyError(f"Feature group '{group_name}' not registered. Call register_feature_group first.")

        feature_names = {f.name for f in group.features} | {group.primary_key, group.event_time_col}
        for rec in records:
            unknown = set(rec.keys()) - feature_names
            if unknown:
                raise ValueError(f"Unknown features {unknown} not in group schema.")

        if group_name not in self._online_store:
            self._online_store[group_name] = {}

        for rec in records:
            key = str(rec.get(group.primary_key))
            if mode == "overwrite":
                self._online_store[group_name][key] = rec
            else:
                existing = self._online_store[group_name].get(key, {})
                existing.update(rec)
                self._online_store[group_name][key] = existing

            if self._redis:
                redis_key = f"fs:{group_name}:{key}"
                self._redis.hset(redis_key, mapping={
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in rec.items()
                })

        logger.info("Wrote %d records to feature group '%s'.", len(records), group_name)

    # ------------------------------------------------------------------
    # Read / Serving
    # ------------------------------------------------------------------

    def get_online_features(
        self,
        group_name: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Low-latency feature lookup for online inference.
        Redis path: O(1) per entity. Memory path: dict lookup.
        """
        results = []
        for eid in entity_ids:
            if self._redis:
                redis_key = f"fs:{group_name}:{eid}"
                raw = self._redis.hgetall(redis_key)
                rec = {k: json.loads(v) if v.startswith(("[", "{")) else v for k, v in raw.items()}
            else:
                rec = self._online_store.get(group_name, {}).get(str(eid), {})

            if feature_names:
                rec = {k: v for k, v in rec.items() if k in feature_names}
            results.append(rec)

        return results

    def get_training_dataset(
        self,
        entity_df: List[Dict[str, Any]],
        feature_groups: List[str],
        label_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Point-in-time correct feature join for training.

        For each entity row (entity_id + timestamp), retrieves features
        that were available AT OR BEFORE that timestamp. Prevents leakage.

        In production: Delta time-travel query:
          SELECT * FROM feature_table
          TIMESTAMP AS OF entity_row.event_time
          WHERE primary_key = entity_row.entity_id
        """
        joined = []
        for entity_row in entity_df:
            combined = dict(entity_row)
            entity_time = entity_row.get("event_time", datetime.now(timezone.utc).isoformat())

            for group_name in feature_groups:
                group = self._registry.get(group_name)
                if not group:
                    logger.warning("Feature group '%s' not found — skipping.", group_name)
                    continue

                entity_id = str(entity_row.get(group.primary_key, ""))
                group_data = self._online_store.get(group_name, {})
                features = group_data.get(entity_id, {})

                # Point-in-time filter: only use features where event_time <= entity_time
                feat_time = features.get(group.event_time_col, "")
                if feat_time and feat_time > entity_time:
                    logger.debug(
                        "Skipping future feature for entity %s (feat_time=%s > entity_time=%s)",
                        entity_id, feat_time, entity_time,
                    )
                    continue

                combined.update(features)

            joined.append(combined)

        self._log_lineage(entity_df, feature_groups, label_col)
        logger.info("Generated training dataset: %d rows, %d feature groups.", len(joined), len(feature_groups))
        return joined

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def _log_lineage(
        self,
        entity_df: List[Dict],
        feature_groups: List[str],
        label_col: Optional[str],
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_entities": len(entity_df),
            "feature_groups": {
                g: self._registry[g].to_dict()
                for g in feature_groups
                if g in self._registry
            },
            "label_col": label_col,
        }
        self._lineage.append(entry)

        if self._mlflow:
            with self._mlflow.start_run(run_name="feature-store-training-dataset"):
                self._mlflow.log_params({
                    "n_entities": len(entity_df),
                    "feature_groups": ",".join(feature_groups),
                    "label_col": label_col or "none",
                })

    def get_lineage(self) -> List[Dict]:
        return self._lineage

    def feature_importance_report(self, group_name: str) -> Dict[str, Any]:
        """Basic feature stats for data quality monitoring."""
        group = self._registry.get(group_name)
        if not group:
            return {}
        records = list(self._online_store.get(group_name, {}).values())
        if not records:
            return {"group": group_name, "n_records": 0}

        report: Dict[str, Any] = {"group": group_name, "n_records": len(records)}
        for feat in group.features:
            vals = [r.get(feat.name) for r in records if r.get(feat.name) is not None]
            if not vals:
                report[feat.name] = {"null_rate": 1.0}
                continue
            null_rate = 1 - len(vals) / len(records)
            report[feat.name] = {"null_rate": round(null_rate, 4), "n_non_null": len(vals)}
            if all(isinstance(v, (int, float)) for v in vals):
                report[feat.name]["mean"] = round(sum(vals) / len(vals), 4)
                report[feat.name]["min"] = min(vals)
                report[feat.name]["max"] = max(vals)

        return report


# ---------------------------------------------------------------------------
# Convenience: build a sample RAG feature group
# ---------------------------------------------------------------------------

def build_rag_feature_group() -> FeatureGroup:
    """Standard feature group for RAG query analytics."""
    return FeatureGroup(
        name="rag_query_stats",
        primary_key="user_id",
        event_time_col="event_time",
        description="Per-user RAG query statistics for personalisation and quality monitoring.",
        features=[
            FeatureSchema("avg_retrieval_latency_ms", "float", description="Rolling avg retrieval latency"),
            FeatureSchema("avg_ragas_faithfulness", "float", description="Rolling avg faithfulness score"),
            FeatureSchema("avg_ragas_relevancy", "float", description="Rolling avg answer relevancy"),
            FeatureSchema("query_count_7d", "int", description="Query count in last 7 days"),
            FeatureSchema("preferred_top_k", "int", description="Preferred top-k from feedback"),
            FeatureSchema("avg_chunk_score", "float", description="Avg retrieved chunk relevance"),
            FeatureSchema("rerank_improvement_rate", "float", description="Fraction of queries where rerank improved result"),
            FeatureSchema("domain_affinity", "string", description="Primary query domain (tech/finance/science)"),
        ],
        tags={"domain": "rag", "team": "llmops"},
    )


if __name__ == "__main__":
    import json

    store = FeatureStore()

    group = build_rag_feature_group()
    store.register_feature_group(
        name=group.name,
        features=group.features,
        primary_key=group.primary_key,
        event_time_col=group.event_time_col,
        description=group.description,
        tags=group.tags,
    )

    store.write_features("rag_query_stats", [
        {"user_id": "u001", "event_time": "2024-01-15T10:00:00Z",
         "avg_retrieval_latency_ms": 42.3, "avg_ragas_faithfulness": 0.847,
         "avg_ragas_relevancy": 0.823, "query_count_7d": 23,
         "preferred_top_k": 5, "avg_chunk_score": 0.78,
         "rerank_improvement_rate": 0.61, "domain_affinity": "tech"},
        {"user_id": "u002", "event_time": "2024-01-15T11:00:00Z",
         "avg_retrieval_latency_ms": 38.1, "avg_ragas_faithfulness": 0.791,
         "avg_ragas_relevancy": 0.801, "query_count_7d": 7,
         "preferred_top_k": 3, "avg_chunk_score": 0.71,
         "rerank_improvement_rate": 0.52, "domain_affinity": "finance"},
    ])

    online = store.get_online_features("rag_query_stats", ["u001", "u002"])
    print("Online features:", json.dumps(online, indent=2))

    entity_df = [
        {"user_id": "u001", "event_time": "2024-01-15T12:00:00Z", "label": 1},
        {"user_id": "u002", "event_time": "2024-01-15T12:00:00Z", "label": 0},
    ]
    training = store.get_training_dataset(entity_df, feature_groups=["rag_query_stats"], label_col="label")
    print("\nTraining dataset:")
    for row in training:
        print(" ", {k: v for k, v in row.items() if k != "event_time"})

    report = store.feature_importance_report("rag_query_stats")
    print("\nFeature report:", json.dumps(report, indent=2))
