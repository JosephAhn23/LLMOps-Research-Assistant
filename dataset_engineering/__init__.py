"""
dataset_engineering/
----------------------
Production dataset engineering for LLM pipelines:

  - DatasetVersion   : DVC-backed versioning with lineage tracking
  - QualityChecker   : schema validation, dedup, drift detection
  - SyntheticGen     : LLM-powered synthetic QA pair generation
  - FeatureStore     : lightweight feature registry with versioning
"""

from dataset_engineering.versioning import DatasetVersion, DatasetLineage
from dataset_engineering.quality import QualityChecker, QualityReport, QualityIssue
from dataset_engineering.synthetic import SyntheticQAGenerator, SyntheticDataset, SyntheticQA
from dataset_engineering.feature_store import FeatureStore, FeatureSpec, FeatureVector

# Backwards-compatible aliases
DataQualityChecker = QualityChecker
SyntheticDataGenerator = SyntheticQAGenerator

__all__ = [
    "DatasetVersion",
    "DatasetLineage",
    "QualityChecker",
    "QualityReport",
    "QualityIssue",
    "SyntheticQAGenerator",
    "SyntheticDataset",
    "SyntheticQA",
    "FeatureStore",
    "FeatureSpec",
    "FeatureVector",
    # aliases
    "DataQualityChecker",
    "SyntheticDataGenerator",
]
