"""
Databricks Notebook — LLMOps Feature Engineering & Model Training
=================================================================
Designed to run as a Databricks notebook (cell-by-cell) or as a standalone
script against a Databricks cluster via `databricks-connect` or `dbx`.

Covers:
  - Databricks Runtime (DBR) session setup
  - Unity Catalog: reading/writing managed tables in a three-level namespace
  - Delta Lake feature store (read from Unity Catalog, write back as managed table)
  - MLflow Managed Tracking (auto-configured on Databricks; no URI needed)
  - Reuses SparkFeatureEngineer, SparkMLClassifier, SparkDocumentClusterer
    from spark_ml/spark_ml_pipeline.py — no logic duplication

Run on Databricks:
    1. Upload this file to your Databricks workspace.
    2. Attach to a cluster with DBR 14.x ML (includes Delta, MLflow, DBFS).
    3. Set CATALOG, SCHEMA, and VOLUME_PATH widgets or env vars.
    4. Run All.

Run locally with databricks-connect:
    databricks-connect configure   # point to your Databricks workspace
    python spark_ml/databricks_notebook.py
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# CELL 1 — Configuration (widgets on Databricks, env vars locally)
# ---------------------------------------------------------------------------

# On Databricks: dbutils.widgets.text("catalog", "llmops")
CATALOG = os.getenv("DATABRICKS_CATALOG", "llmops")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "research_assistant")
VOLUME_PATH = os.getenv("DATABRICKS_VOLUME_PATH", f"/Volumes/{CATALOG}/{SCHEMA}/raw_data")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", f"/Users/llmops/spark-ml-pipeline")

# Unity Catalog three-level namespace: catalog.schema.table
RAW_TABLE = f"{CATALOG}.{SCHEMA}.raw_documents"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.document_features"
CLUSTER_TABLE = f"{CATALOG}.{SCHEMA}.document_clusters"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.query_intent_classifier"


# ---------------------------------------------------------------------------
# CELL 2 — Spark session (uses existing cluster on Databricks; local fallback)
# ---------------------------------------------------------------------------

def get_spark():
    """
    On Databricks, `spark` is injected automatically.
    Locally (databricks-connect or plain PySpark), we build a session.
    """
    try:
        # Databricks injects `spark` into the global namespace
        return spark  # noqa: F821
    except NameError:
        pass

    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName("LLMOps-Databricks")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


spark = get_spark()
spark.sparkContext.setLogLevel("WARN")
logger.info("Spark version: %s", spark.version)


# ---------------------------------------------------------------------------
# CELL 3 — Unity Catalog setup
# ---------------------------------------------------------------------------

def setup_unity_catalog(spark, catalog: str, schema: str) -> None:
    """
    Create the catalog and schema if they don't exist.
    On Databricks with Unity Catalog enabled, this is the standard pattern.
    Falls back gracefully if the caller lacks CREATE CATALOG privilege.
    """
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        logger.info("Catalog ready: %s", catalog)
    except Exception as e:
        logger.warning("Could not create catalog (may already exist or lack privilege): %s", e)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    logger.info("Schema ready: %s.%s", catalog, schema)

    # Set default catalog + schema for the session so bare table names resolve
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
    logger.info("Active namespace: %s.%s", catalog, schema)


setup_unity_catalog(spark, CATALOG, SCHEMA)


# ---------------------------------------------------------------------------
# CELL 4 — Load or generate raw data
# ---------------------------------------------------------------------------

def load_or_create_raw_data(spark, table: str, volume_path: str):
    """
    Try to read the raw documents table from Unity Catalog.
    If it doesn't exist (first run), create a synthetic dataset and register it.
    In production, replace the synthetic data with a real ingestion pipeline.
    """
    try:
        df = spark.read.table(table)
        logger.info("Loaded %d rows from %s", df.count(), table)
        return df
    except Exception:
        logger.info("Table %s not found — creating synthetic dataset", table)

    data = [
        ("What is retrieval-augmented generation?", "retrieval"),
        ("How does FAISS vector search work?", "retrieval"),
        ("Explain cross-encoder reranking", "retrieval"),
        ("Generate an image of a mountain at sunset", "generation"),
        ("Create a diagram of the transformer architecture", "generation"),
        ("Draw a flowchart for the RAG pipeline", "generation"),
        ("Summarise this research paper on LLMs", "summarisation"),
        ("What are the key findings of the RLHF paper?", "summarisation"),
        ("Give me a brief overview of attention mechanisms", "summarisation"),
        ("Compare RAG vs fine-tuning for domain adaptation", "analysis"),
        ("What are the trade-offs between FAISS and Pinecone?", "analysis"),
        ("Evaluate the quality of this generated response", "evaluation"),
    ] * 50  # repeat for a realistic dataset size

    df = spark.createDataFrame(data, ["text", "intent"])

    # Write as a managed Delta table in Unity Catalog
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    logger.info("Created and registered %s (%d rows)", table, df.count())
    return df


df_raw = load_or_create_raw_data(spark, RAW_TABLE, VOLUME_PATH)
df_raw.show(5, truncate=True)


# ---------------------------------------------------------------------------
# CELL 5 — Feature engineering (reuses SparkFeatureEngineer)
# ---------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spark_ml.spark_ml_pipeline import SparkFeatureEngineer, SparkMLConfig

config = SparkMLConfig(
    app_name="LLMOps-Databricks",
    master="local[*]",  # ignored on Databricks (cluster handles this)
    mlflow_uri="databricks",  # Databricks managed MLflow — no URI needed
    mlflow_experiment=MLFLOW_EXPERIMENT,
    feature_store_path=FEATURE_TABLE,
)

engineer = SparkFeatureEngineer(spark, config)

# Encode intent labels as numeric for MLlib
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="intent", outputCol="label")
indexer_model = indexer.fit(df_raw)
df_indexed = indexer_model.transform(df_raw)
label_map = dict(enumerate(indexer_model.labels))
logger.info("Label mapping: %s", label_map)

# Build TF-IDF features
df_features, tfidf_model = engineer.build_tfidf_features(df_indexed, text_col="text")
df_features = engineer.build_quality_features(df_features, text_col="text")

logger.info("Feature schema: %s", df_features.schema.fieldNames())


# ---------------------------------------------------------------------------
# CELL 6 — Write feature table to Unity Catalog (Delta)
# ---------------------------------------------------------------------------

def write_feature_table(df, table: str) -> None:
    """
    Persist the feature DataFrame as a managed Delta table in Unity Catalog.
    This acts as the feature store — downstream training jobs read from here.
    """
    # Drop non-serialisable ML vector columns before saving
    # (keep raw features; vectors can be recomputed from the pipeline model)
    cols_to_save = [c for c in df.columns if c not in ("tf", "tfidf")]
    (
        df.select(cols_to_save)
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    logger.info("Feature table written: %s", table)


write_feature_table(df_features, FEATURE_TABLE)


# ---------------------------------------------------------------------------
# CELL 7 — Train GBT query intent classifier (reuses SparkMLClassifier)
# ---------------------------------------------------------------------------

from spark_ml.spark_ml_pipeline import SparkMLClassifier
import mlflow

# On Databricks, set the experiment path (workspace path, not a URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

classifier = SparkMLClassifier(spark, config)
model, f1 = classifier.train(df_features, label_col="label", feature_col="features")
logger.info("GBT F1 score: %.4f", f1)


# ---------------------------------------------------------------------------
# CELL 8 — Register model in Unity Catalog Model Registry
# ---------------------------------------------------------------------------

def register_model_unity_catalog(
    model_uri: str,
    registered_name: str,
    description: str = "",
) -> str:
    """
    Register a model in the Unity Catalog Model Registry.
    Three-level name: catalog.schema.model_name
    Returns the model version string.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        client.create_registered_model(
            registered_name,
            description=description,
        )
        logger.info("Created registered model: %s", registered_name)
    except Exception:
        logger.info("Registered model already exists: %s", registered_name)

    mv = client.create_model_version(
        name=registered_name,
        source=model_uri,
        description=description,
    )
    logger.info(
        "Registered model version %s: %s (run_id=%s)",
        mv.version, registered_name, mv.run_id,
    )
    return mv.version


# Get the latest run's model URI and register it
runs = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT],
    order_by=["start_time DESC"],
    max_results=1,
)
if not runs.empty:
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/spark-gbt-model"
    version = register_model_unity_catalog(
        model_uri=model_uri,
        registered_name=MODEL_NAME,
        description="GBT query intent classifier — routes queries to retrieval strategy",
    )
    logger.info("Model registered as %s version %s", MODEL_NAME, version)
else:
    logger.warning("No MLflow runs found — skipping model registration")


# ---------------------------------------------------------------------------
# CELL 9 — KMeans topic clustering (reuses SparkDocumentClusterer)
# ---------------------------------------------------------------------------

from spark_ml.spark_ml_pipeline import SparkDocumentClusterer

clusterer = SparkDocumentClusterer(spark, config)
optimal_k = clusterer.find_optimal_k(df_features, k_range=range(3, 8))
df_clustered, kmeans_model, silhouette = clusterer.cluster(df_features, k=optimal_k)
logger.info("Optimal k=%d, silhouette=%.4f", optimal_k, silhouette)


# ---------------------------------------------------------------------------
# CELL 10 — Write cluster assignments to Unity Catalog
# ---------------------------------------------------------------------------

(
    df_clustered.select("text", "intent", "label", "prediction")
    .withColumnRenamed("prediction", "cluster_id")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CLUSTER_TABLE)
)
logger.info("Cluster table written: %s", CLUSTER_TABLE)


# ---------------------------------------------------------------------------
# CELL 11 — Summary query (shows Unity Catalog tables)
# ---------------------------------------------------------------------------

logger.info("=== Unity Catalog Tables ===")
spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").show()

logger.info("=== Feature Table Sample ===")
spark.read.table(FEATURE_TABLE).select("text", "intent", "char_count", "word_count").show(5)

logger.info("=== Cluster Distribution ===")
spark.read.table(CLUSTER_TABLE).groupBy("cluster_id").count().orderBy("cluster_id").show()

logger.info("Databricks notebook complete.")
logger.info("Model registered: %s", MODEL_NAME)
logger.info("Feature table: %s", FEATURE_TABLE)
logger.info("Cluster table: %s", CLUSTER_TABLE)
