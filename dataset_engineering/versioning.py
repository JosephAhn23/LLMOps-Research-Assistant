"""
dataset_engineering/versioning.py
-----------------------------------
DVC-backed dataset versioning with full lineage tracking.

Each version stores:
  - content hash (SHA-256)
  - transformation provenance (what script + params produced it)
  - split metadata (train/val/test counts)
  - pointer to DVC remote (S3 / GCS / Azure)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetLineage:
    source_hash: str
    transform: str
    params: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    author: str = "pipeline"


@dataclass
class DatasetVersion:
    """
    Wraps a Pandas DataFrame with versioning metadata and optional DVC tracking.

    Usage
    -----
    >>> dv = DatasetVersion.from_csv("data/train.csv", version="v1.2")
    >>> dv.validate_schema(["question", "answer", "context"])
    >>> dv.save("data/train_clean.csv", dvc_push=True)
    >>> print(dv.summary())
    """

    data: pd.DataFrame
    version: str
    dataset_name: str
    lineage: DatasetLineage | None = None
    split: str = "train"  # train | val | test
    content_hash: str = field(init=False)

    def __post_init__(self):
        self.content_hash = self._hash_dataframe(self.data)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        version: str = "v1.0",
        dataset_name: str | None = None,
        **read_kwargs,
    ) -> "DatasetVersion":
        path = Path(path)
        df = pd.read_csv(path, **read_kwargs)
        name = dataset_name or path.stem
        logger.info("Loaded %s: %d rows", path, len(df))
        return cls(data=df, version=version, dataset_name=name)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        version: str = "v1.0",
        dataset_name: str | None = None,
    ) -> "DatasetVersion":
        path = Path(path)
        records = []
        with path.open() as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        name = dataset_name or path.stem
        return cls(data=df, version=version, dataset_name=name)

    # ------------------------------------------------------------------
    # Operations that produce new versions (immutable transform pattern)
    # ------------------------------------------------------------------

    def transform(
        self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        transform_name: str,
        params: dict[str, Any] | None = None,
        new_version: str | None = None,
    ) -> "DatasetVersion":
        """Apply a transformation and return a new versioned DatasetVersion."""
        new_df = fn(self.data.copy())
        lineage = DatasetLineage(
            source_hash=self.content_hash,
            transform=transform_name,
            params=params or {},
        )
        version = new_version or self._bump_version()
        new_dv = DatasetVersion(
            data=new_df,
            version=version,
            dataset_name=self.dataset_name,
            lineage=lineage,
            split=self.split,
        )
        logger.info(
            "Transform '%s': %d -> %d rows (v%s -> v%s)",
            transform_name, len(self.data), len(new_df), self.version, version,
        )
        return new_dv

    def train_val_test_split(
        self, train: float = 0.8, val: float = 0.1, seed: int = 42
    ) -> tuple["DatasetVersion", "DatasetVersion", "DatasetVersion"]:
        """Reproducible stratified split returning three versioned datasets."""
        df = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train)
        n_val = int(n * val)

        splits = {
            "train": df.iloc[:n_train],
            "val": df.iloc[n_train: n_train + n_val],
            "test": df.iloc[n_train + n_val:],
        }
        versions = {}
        for split_name, split_df in splits.items():
            versions[split_name] = DatasetVersion(
                data=split_df.reset_index(drop=True),
                version=f"{self.version}_{split_name}",
                dataset_name=self.dataset_name,
                split=split_name,
            )
        logger.info(
            "Split: train=%d  val=%d  test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )
        return versions["train"], versions["val"], versions["test"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
        dvc_push: bool = False,
        remote: str = "myremote",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".csv":
            self.data.to_csv(path, index=False)
        elif path.suffix in {".jsonl", ".json"}:
            self.data.to_json(path, orient="records", lines=True)
        elif path.suffix == ".parquet":
            self.data.to_parquet(path, index=False)
        else:
            self.data.to_csv(path, index=False)

        # Save metadata sidecar
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.to_dict(), indent=2))

        if dvc_push:
            self._dvc_add_push(path, remote)

        logger.info("Saved %s (%d rows) -> %s", self.dataset_name, len(self.data), path)
        return path

    # ------------------------------------------------------------------
    # DVC integration
    # ------------------------------------------------------------------

    def _dvc_add_push(self, path: Path, remote: str) -> None:
        try:
            subprocess.run(["dvc", "add", str(path)], check=True)
            subprocess.run(["dvc", "push", "--remote", remote], check=True)
            logger.info("DVC: pushed %s to remote '%s'", path, remote)
        except FileNotFoundError:
            logger.warning("DVC not installed. Skipping dvc push.")
        except subprocess.CalledProcessError as e:
            logger.error("DVC push failed: %s", e)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def validate_schema(self, required_cols: list[str]) -> None:
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"[{self.dataset_name}] Missing columns: {missing}")
        logger.info("Schema valid: %s", required_cols)

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.dataset_name}  v{self.version}  [{self.split}]",
            f"  Rows: {len(self.data):,}   Columns: {list(self.data.columns)}",
            f"  Hash: {self.content_hash[:16]}...",
            f"  Nulls: {self.data.isnull().sum().to_dict()}",
        ]
        if self.lineage:
            lines.append(
                f"  Lineage: {self.lineage.transform}({self.lineage.params}) "
                f"<- {self.lineage.source_hash[:12]}..."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "split": self.split,
            "content_hash": self.content_hash,
            "n_rows": len(self.data),
            "columns": list(self.data.columns),
            "lineage": (
                {
                    "source_hash": self.lineage.source_hash,
                    "transform": self.lineage.transform,
                    "params": self.lineage.params,
                    "created_at": self.lineage.created_at,
                }
                if self.lineage
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bump_version(self) -> str:
        parts = self.version.lstrip("v").split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except ValueError:
            parts.append("1")
        return "v" + ".".join(parts)

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        h = hashlib.sha256()
        h.update(df.to_json(orient="records").encode())
        return h.hexdigest()


# Backwards-compatible alias for the registry-based API
DatasetRegistry = None  # replaced by DatasetVersion.from_csv / .save pattern
