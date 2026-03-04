"""
Spark ML Pipeline — full ML training on top of Spark ingestion.

Extends ingestion/spark_ml_pipeline.py (feature engineering) into
end-to-end ML training:

  - SparkFeatureEngineer  — TF-IDF + sentence-transformer pandas_udf embeddings +
                            quality features (char count, avg word length,
                            punct density, unique word ratio) via VectorAssembler
  - SparkMLClassifier     — GBT query intent classifier with MLflow Spark tracking
                            (routes queries to correct retrieval strategy)
  - SparkDocumentClusterer — KMeans topic discovery with elbow method
  - DeltaFeatureStore     — Delta Lake feature store with Parquet fallback

Covers gaps: Spark ML, GBT classifier, MLlib, MLflow + Spark, Delta Lake,
             pandas_udf, distributed feature engineering.

Run locally:
    python spark_ml/spark_ml_pipeline.py
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SparkMLConfig:
    app_name: str = "LLMOps-SparkML"
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "8g"
    shuffle_partitions: int = 50
    # Feature engineering
    tfidf_num_features: int = 2 ** 16
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    # GBT classifier
    num_trees: int = 50
    max_depth: int = 5
    max_bins: int = 32
    # KMeans
    min_clusters: int = 3
    max_clusters: int = 10
    kmeans_max_iter: int = 20
    kmeans_seed: int = 42
    # MLflow
    mlflow_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "spark-ml-pipeline"
    # Storage
    feature_store_path: str = "delta/feature_store"
    model_output_path: str = "spark_ml_output"


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def build_spark_session(config: SparkMLConfig):
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.executor.memory", config.executor_memory)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

class SparkFeatureEngineer:
    """
    Distributed feature engineering on raw text corpora.

    Produces a feature store compatible with MLlib classifiers:
      - TF-IDF (HashingTF → IDF → L2 normalisation)
      - Sentence-transformer embeddings via pandas_udf (vectorised batch execution)
      - Quality features: char_count, word_count, avg_word_len,
                          punct_density, unique_word_ratio
      - VectorAssembler combines quality features into a single vector
    """

    def __init__(self, spark, config: SparkMLConfig):
        self.spark = spark
        self.config = config

    def build_tfidf_features(self, df, text_col: str = "text"):
        """TF-IDF vectorisation pipeline. Returns (transformed_df, fitted_pipeline_model)."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            HashingTF, IDF, Normalizer, RegexTokenizer, StopWordsRemover,
        )

        stages = [
            RegexTokenizer(inputCol=text_col, outputCol="tokens_raw", pattern=r"\W+"),
            StopWordsRemover(inputCol="tokens_raw", outputCol="tokens"),
            HashingTF(inputCol="tokens", outputCol="tf",
                      numFeatures=self.config.tfidf_num_features),
            IDF(inputCol="tf", outputCol="tfidf", minDocFreq=5),
            Normalizer(inputCol="tfidf", outputCol="features", p=2.0),
        ]
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        return model.transform(df), model

    def build_embedding_features(self, df, text_col: str = "text"):
        """
        Compute sentence embeddings via a pandas_udf wrapping sentence-transformers.
        pandas_udf executes in vectorised batches on each executor — significantly
        faster than a row-by-row Python UDF.
        """
        import pandas as pd
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import ArrayType, FloatType

        embedding_model_name = self.config.embedding_model

        @pandas_udf(ArrayType(FloatType()))
        def embed_batch(texts: pd.Series) -> pd.Series:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embedding_model_name)
            embeddings = model.encode(
                texts.tolist(), batch_size=32, show_progress_bar=False
            )
            return pd.Series([emb.tolist() for emb in embeddings])

        return df.withColumn("embedding", embed_batch(df[text_col]))

    def build_quality_features(self, df, text_col: str = "text"):
        """
        Compute text quality signals as ML features:
          char_count, word_count, avg_word_len, punct_density, unique_word_ratio
        """
        from pyspark.sql import functions as F
        from pyspark.ml.feature import VectorAssembler

        df = (
            df.withColumn("char_count", F.length(F.col(text_col)).cast("float"))
            .withColumn("word_count", F.size(F.split(F.col(text_col), r"\s+")).cast("float"))
            .withColumn(
                "avg_word_len",
                (F.col("char_count") / (F.col("word_count") + 1e-6)).cast("float"),
            )
            .withColumn(
                "punct_density",
                (
                    F.length(F.regexp_replace(F.col(text_col), r"[^.,!?;:]", "")).cast("float")
                    / (F.col("char_count") + 1e-6)
                ),
            )
            .withColumn(
                "unique_word_ratio",
                (
                    F.size(
                        F.array_distinct(F.split(F.lower(F.col(text_col)), r"\s+"))
                    ).cast("float")
                    / (F.col("word_count") + 1e-6)
                ),
            )
        )

        assembler = VectorAssembler(
            inputCols=[
                "char_count", "word_count", "avg_word_len",
                "punct_density", "unique_word_ratio",
            ],
            outputCol="quality_features",
            handleInvalid="keep",
        )
        return assembler.transform(df)

    def save_feature_store(self, df, path: Optional[str] = None) -> None:
        """Persist features as a Delta Lake table."""
        dest = path or self.config.feature_store_path
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(dest)
        )
        logger.info("Feature store written to %s", dest)


# ---------------------------------------------------------------------------
# GBT Query Intent Classifier
# ---------------------------------------------------------------------------

class SparkMLClassifier:
    """
    GBT (Gradient Boosted Trees) query intent classifier.

    Classifies queries into intent categories to route each query to the
    optimal retrieval strategy. Tracks training metrics and logs the Spark
    model artifact in MLflow.
    """

    INTENT_LABELS = [
        "factual_lookup", "reasoning", "summarisation",
        "code_generation", "comparison",
    ]

    def __init__(self, spark, config: SparkMLConfig):
        self.spark = spark
        self.config = config
        self.model = None

    def _make_synthetic_dataset(self):
        """Create a labelled synthetic dataset for demo/testing."""
        import random
        from pyspark.sql import Row

        random.seed(42)
        templates = {
            0: ["What is {}", "Define {}", "Who invented {}", "When was {} created"],
            1: ["Why does {} happen", "How does {} work", "Explain the reasoning behind {}"],
            2: ["Summarise {} in one paragraph", "Give me a brief overview of {}"],
            3: ["Write a function to {}", "Implement {} in Python", "Show me code for {}"],
            4: ["Compare {} and {}", "Difference between {} and {}", "Pros and cons of {}"],
        }
        topics = ["RAG", "FAISS", "transformers", "LLMs", "fine-tuning", "RLHF", "embeddings"]
        rows = []
        for label, tmps in templates.items():
            for _ in range(40):
                t = random.choice(tmps)
                topic = random.choice(topics)
                text = t.format(topic, random.choice(topics))
                rows.append(Row(text=text, label=float(label)))
        return self.spark.createDataFrame(rows)

    def train(self, df=None, label_col: str = "label", feature_col: str = "features") -> Dict:
        """
        Train a GBT classifier. Uses synthetic data if df is None.
        Logs params, metrics, and the Spark model artifact to MLflow.
        """
        import mlflow
        import mlflow.spark
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        from pyspark.ml.feature import (
            HashingTF, IDF, RegexTokenizer, StopWordsRemover,
        )

        if df is None:
            df = self._make_synthetic_dataset()

        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens_raw", pattern=r"\W+")
        remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
        tf = HashingTF(inputCol="tokens", outputCol="tf",
                       numFeatures=self.config.tfidf_num_features)
        idf = IDF(inputCol="tf", outputCol=feature_col, minDocFreq=1)
        gbt = GBTClassifier(
            featuresCol=feature_col,
            labelCol=label_col,
            maxIter=self.config.num_trees,
            maxDepth=self.config.max_depth,
            maxBins=self.config.max_bins,
            seed=42,
        )
        pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, gbt])

        train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        with mlflow.start_run(run_name="gbt-query-intent"):
            mlflow.log_params({
                "maxIter": self.config.num_trees,
                "maxDepth": self.config.max_depth,
                "maxBins": self.config.max_bins,
            })
            self.model = pipeline.fit(train_df)
            preds = self.model.transform(val_df)

            acc_eval = MulticlassClassificationEvaluator(
                labelCol=label_col, predictionCol="prediction", metricName="accuracy"
            )
            f1_eval = MulticlassClassificationEvaluator(
                labelCol=label_col, predictionCol="prediction", metricName="f1"
            )
            accuracy = acc_eval.evaluate(preds)
            f1 = f1_eval.evaluate(preds)

            mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
            mlflow.spark.log_model(self.model, artifact_path="spark-gbt-model")
            logger.info("GBT classifier: accuracy=%.3f f1=%.3f", accuracy, f1)

        return {"accuracy": accuracy, "f1": f1}

    def predict(self, df, feature_col: str = "features"):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.transform(df)

    def save(self, path: Optional[str] = None) -> None:
        dest = path or os.path.join(self.config.model_output_path, "gbt_classifier")
        if self.model:
            self.model.write().overwrite().save(dest)
            logger.info("Spark ML model saved to %s", dest)

    def load(self, path: Optional[str] = None) -> "SparkMLClassifier":
        from pyspark.ml.classification import GBTClassificationModel
        src = path or os.path.join(self.config.model_output_path, "gbt_classifier")
        self.model = GBTClassificationModel.load(src)
        logger.info("GBT model loaded from %s", src)
        return self


# ---------------------------------------------------------------------------
# KMeans Topic Clusterer
# ---------------------------------------------------------------------------

class SparkDocumentClusterer:
    """
    KMeans clustering on document TF-IDF features for topic discovery.
    Elbow method selects optimal k by maximising silhouette score.
    """

    def __init__(self, spark, config: SparkMLConfig):
        self.spark = spark
        self.config = config
        self._model = None
        self._feat_model = None
        self._k = None

    def _build_features(self, df):
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            HashingTF, IDF, Normalizer, RegexTokenizer, StopWordsRemover,
        )
        pipeline = Pipeline(stages=[
            RegexTokenizer(inputCol="text", outputCol="tokens_raw", pattern=r"\W+"),
            StopWordsRemover(inputCol="tokens_raw", outputCol="tokens"),
            HashingTF(inputCol="tokens", outputCol="tf",
                      numFeatures=self.config.tfidf_num_features),
            IDF(inputCol="tf", outputCol="tfidf", minDocFreq=1),
            Normalizer(inputCol="tfidf", outputCol="features", p=2.0),
        ])
        model = pipeline.fit(df)
        return model.transform(df), model

    def find_optimal_k(self, df) -> Tuple[int, Dict[int, float]]:
        """
        Elbow method: fit KMeans for k in [min_clusters, max_clusters].
        Returns (optimal_k, {k: silhouette_score}).
        """
        import mlflow
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.evaluation import ClusteringEvaluator

        df_feat, _ = self._build_features(df)
        evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
        scores: Dict[int, float] = {}

        for k in range(self.config.min_clusters, self.config.max_clusters + 1):
            km = KMeans(
                featuresCol="features", k=k,
                maxIter=self.config.kmeans_max_iter, seed=self.config.kmeans_seed,
            )
            model = km.fit(df_feat)
            score = float(evaluator.evaluate(model.transform(df_feat)))
            scores[k] = score
            logger.info("k=%d | silhouette=%.4f", k, score)

        optimal = max(scores, key=scores.get)
        logger.info("Optimal k=%d (highest silhouette)", optimal)
        return optimal, scores

    def cluster(
        self,
        df,
        k: Optional[int] = None,
        feature_col: str = "features",
    ):
        """
        Fit KMeans with the given k (or auto-select via elbow method).
        Returns (clustered_df, model, silhouette_score).
        """
        import mlflow
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.evaluation import ClusteringEvaluator

        df_feat, self._feat_model = self._build_features(df)
        if k is None:
            k, _ = self.find_optimal_k(df)

        km = KMeans(
            featuresCol=feature_col, k=k,
            maxIter=self.config.kmeans_max_iter, seed=self.config.kmeans_seed,
        )

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run(run_name=f"kmeans-k{k}"):
            self._model = km.fit(df_feat)
            clustered = self._model.transform(df_feat)
            evaluator = ClusteringEvaluator(featuresCol=feature_col)
            silhouette = float(evaluator.evaluate(clustered))
            mlflow.log_params({"k": k, "maxIter": self.config.kmeans_max_iter})
            mlflow.log_metric("silhouette_score", silhouette)
            logger.info("KMeans k=%d silhouette=%.4f", k, silhouette)

        self._k = k
        return clustered, self._model, silhouette

    def save_to_delta(self, df, path: Optional[str] = None) -> None:
        """Persist cluster assignments to Delta Lake (Parquet fallback)."""
        dest = path or os.path.join(self.config.feature_store_path, "document_clusters")
        clustered = self._model.transform(self._feat_model.transform(df))
        try:
            (
                clustered.select("text", "prediction")
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(dest)
            )
            logger.info("Cluster assignments written to Delta Lake: %s", dest)
        except Exception as e:
            logger.warning("Delta write failed — falling back to Parquet: %s", e)
            parquet_dest = dest.replace("delta/", "parquet/")
            clustered.select("text", "prediction").write.mode("overwrite").parquet(parquet_dest)
            logger.info("Cluster assignments written to Parquet: %s", parquet_dest)


# ---------------------------------------------------------------------------
# Delta Lake Feature Store
# ---------------------------------------------------------------------------

class DeltaFeatureStore:
    """
    Persist and retrieve feature vectors from Delta Lake.
    Falls back to Parquet if Delta Lake JARs are not available.
    """

    def __init__(self, base_path: str = "delta/feature_store"):
        self.base_path = base_path

    def write(self, df, table_name: str, mode: str = "overwrite") -> None:
        path = os.path.join(self.base_path, table_name)
        try:
            (
                df.write.format("delta")
                .mode(mode)
                .option("overwriteSchema", "true")
                .save(path)
            )
            logger.info("Feature table '%s' written to Delta Lake: %s", table_name, path)
        except Exception as e:
            logger.warning("Delta write failed — falling back to Parquet: %s", e)
            df.write.mode(mode).parquet(path + "_parquet")

    def read(self, spark, table_name: str):
        path = os.path.join(self.base_path, table_name)
        try:
            return spark.read.format("delta").load(path)
        except Exception:
            return spark.read.parquet(path + "_parquet")

    def upsert(self, spark, new_df, table_name: str, merge_key: str = "id") -> None:
        """Delta Lake MERGE (upsert) — insert new rows, update existing."""
        from delta.tables import DeltaTable
        path = os.path.join(self.base_path, table_name)
        try:
            if DeltaTable.isDeltaTable(spark, path):
                DeltaTable.forPath(spark, path).alias("existing").merge(
                    new_df.alias("updates"),
                    f"existing.{merge_key} = updates.{merge_key}",
                ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
                logger.info("Upserted into Delta table '%s'", table_name)
            else:
                self.write(new_df, table_name)
        except Exception as e:
            logger.warning("Delta upsert failed: %s — writing as overwrite", e)
            self.write(new_df, table_name)


# ---------------------------------------------------------------------------
# Entry point / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = SparkMLConfig(master="local[2]")
    spark = build_spark_session(config)

    data = [
        ("What is RAG?", 0),
        ("How does FAISS work?", 0),
        ("Generate an image of a sunset", 1),
        ("Summarize this paper", 0),
        ("Create a diagram", 1),
        ("Explain attention mechanisms", 0),
        ("Write a Python function for embeddings", 1),
        ("Compare BERT and GPT", 0),
        ("Implement LoRA in PyTorch", 1),
        ("What is RLHF?", 0),
    ] * 10

    df = spark.createDataFrame(data, ["text", "label"])

    engineer = SparkFeatureEngineer(spark, config)
    df_tfidf, feat_model = engineer.build_tfidf_features(df)
    df_quality = engineer.build_quality_features(df)
    logger.info("TF-IDF schema: %s", df_tfidf.schema.fieldNames())
    logger.info("Quality schema: %s", df_quality.schema.fieldNames())

    classifier = SparkMLClassifier(spark, config)
    metrics = classifier.train()
    logger.info("Classifier metrics: %s", metrics)

    clusterer = SparkDocumentClusterer(spark, config)
    clustered, km_model, silhouette = clusterer.cluster(df, k=3)
    clustered.groupBy("prediction").count().orderBy("prediction").show()

    logger.info("Spark ML pipeline smoke test complete.")
    spark.stop()
