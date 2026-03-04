"""
Spark ML (MLlib) feature engineering and preprocessing pipeline.

Extends the base Spark ingestion pipeline with MLlib transformers:
  - TF-IDF feature extraction (HashingTF + IDF)
  - Word2Vec embeddings for document-level representations
  - Normalisation, StandardScaler
  - PCA dimensionality reduction
  - K-Means clustering for corpus analysis
  - ML Pipeline API (fit/transform) for reproducible preprocessing

Covers gaps: Spark ML, MLlib, distributed feature engineering,
             ML Pipeline API, cluster-based corpus analysis.
"""
from __future__ import annotations

import logging
from typing import Optional

from pyspark.ml import Pipeline as SparkMLPipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    Normalizer,
    PCA,
    RegexTokenizer,
    StopWordsRemover,
    StringIndexer,
    VectorAssembler,
    Word2Vec,
)
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

logger = logging.getLogger(__name__)


def get_spark(app_name: str = "LLMOpsSparkML") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# TF-IDF feature pipeline
# ---------------------------------------------------------------------------


class TFIDFFeaturePipeline:
    """
    Distributed TF-IDF feature extraction using Spark MLlib.

    Stages:
      RegexTokenizer → StopWordsRemover → HashingTF → IDF → Normalizer
    """

    def __init__(
        self,
        input_col: str = "text_clean",
        output_col: str = "tfidf_features",
        num_features: int = 65536,
        min_doc_freq: int = 2,
    ):
        self.input_col = input_col
        self.output_col = output_col
        self.num_features = num_features
        self.min_doc_freq = min_doc_freq
        self._pipeline: Optional[SparkMLPipeline] = None
        self._fitted = None

    def build(self) -> SparkMLPipeline:
        tokenizer = RegexTokenizer(
            inputCol=self.input_col,
            outputCol="tokens",
            pattern=r"\W+",
            minTokenLength=2,
        )
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        hashing_tf = HashingTF(
            inputCol="filtered_tokens",
            outputCol="raw_features",
            numFeatures=self.num_features,
        )
        idf = IDF(
            inputCol="raw_features",
            outputCol="idf_features",
            minDocFreq=self.min_doc_freq,
        )
        normalizer = Normalizer(inputCol="idf_features", outputCol=self.output_col, p=2.0)

        self._pipeline = SparkMLPipeline(
            stages=[tokenizer, remover, hashing_tf, idf, normalizer]
        )
        return self._pipeline

    def fit_transform(self, df: DataFrame) -> DataFrame:
        if self._pipeline is None:
            self.build()
        self._fitted = self._pipeline.fit(df)
        return self._fitted.transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        if self._fitted is None:
            raise RuntimeError("Call fit_transform first to fit the pipeline.")
        return self._fitted.transform(df)

    def save(self, path: str):
        if self._fitted is None:
            raise RuntimeError("Nothing to save — pipeline not fitted.")
        self._fitted.write().overwrite().save(path)
        logger.info("TF-IDF pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TFIDFFeaturePipeline":
        from pyspark.ml import PipelineModel

        obj = cls()
        obj._fitted = PipelineModel.load(path)
        return obj


# ---------------------------------------------------------------------------
# Word2Vec document embeddings
# ---------------------------------------------------------------------------


class Word2VecEmbeddingPipeline:
    """
    Distributed Word2Vec embeddings via Spark MLlib.

    Produces a fixed-size document vector by averaging word embeddings.
    Useful as a lightweight dense representation for clustering or
    downstream ML without requiring a GPU.
    """

    def __init__(
        self,
        input_col: str = "text_clean",
        output_col: str = "w2v_embedding",
        vector_size: int = 128,
        min_count: int = 3,
        num_partitions: int = 8,
        max_iter: int = 5,
        window_size: int = 5,
    ):
        self.input_col = input_col
        self.output_col = output_col
        self.vector_size = vector_size
        self.min_count = min_count
        self.num_partitions = num_partitions
        self.max_iter = max_iter
        self.window_size = window_size
        self._fitted = None

    def fit_transform(self, df: DataFrame) -> DataFrame:
        tokenizer = RegexTokenizer(
            inputCol=self.input_col, outputCol="_w2v_tokens",
            pattern=r"\W+", minTokenLength=2,
        )
        remover = StopWordsRemover(inputCol="_w2v_tokens", outputCol="_w2v_filtered")
        w2v = Word2Vec(
            inputCol="_w2v_filtered",
            outputCol=self.output_col,
            vectorSize=self.vector_size,
            minCount=self.min_count,
            numPartitions=self.num_partitions,
            maxIter=self.max_iter,
            windowSize=self.window_size,
            seed=42,
        )
        pipeline = SparkMLPipeline(stages=[tokenizer, remover, w2v])
        self._fitted = pipeline.fit(df)
        result = self._fitted.transform(df)
        logger.info(
            "Word2Vec fitted — vector_size=%d  min_count=%d",
            self.vector_size, self.min_count,
        )
        return result

    def find_synonyms(self, word: str, num: int = 10) -> list[tuple[str, float]]:
        """Return the top-N synonyms for a word using the fitted model."""
        if self._fitted is None:
            raise RuntimeError("Call fit_transform first.")
        w2v_model = self._fitted.stages[-1]
        return w2v_model.findSynonyms(word, num).collect()


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------


class SparkPCAReducer:
    """
    Distributed PCA via Spark MLlib.

    Reduces high-dimensional TF-IDF or W2V vectors to a lower-dimensional
    space for visualisation or downstream clustering.
    """

    def __init__(self, input_col: str, output_col: str = "pca_features", k: int = 50):
        self.input_col = input_col
        self.output_col = output_col
        self.k = k
        self._fitted = None

    def fit_transform(self, df: DataFrame) -> DataFrame:
        pca = PCA(k=self.k, inputCol=self.input_col, outputCol=self.output_col)
        self._fitted = pca.fit(df)
        explained = sum(self._fitted.explainedVariance.toArray())
        logger.info(
            "PCA fitted — k=%d  explained_variance=%.3f", self.k, explained
        )
        return self._fitted.transform(df)


# ---------------------------------------------------------------------------
# K-Means corpus clustering
# ---------------------------------------------------------------------------


class CorpusClusteringPipeline:
    """
    K-Means clustering over document embeddings using Spark MLlib.

    Useful for:
      - Corpus analysis and topic discovery
      - Balanced data sampling across semantic clusters
      - Detecting data drift between train and eval splits
    """

    def __init__(
        self,
        feature_col: str = "tfidf_features",
        k: int = 20,
        max_iter: int = 30,
        seed: int = 42,
    ):
        self.feature_col = feature_col
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self._model = None

    def fit(self, df: DataFrame) -> "CorpusClusteringPipeline":
        kmeans = KMeans(
            featuresCol=self.feature_col,
            predictionCol="cluster_id",
            k=self.k,
            maxIter=self.max_iter,
            seed=self.seed,
        )
        self._model = kmeans.fit(df)
        evaluator = ClusteringEvaluator(
            featuresCol=self.feature_col, predictionCol="cluster_id"
        )
        predictions = self._model.transform(df)
        silhouette = evaluator.evaluate(predictions)
        logger.info(
            "K-Means fitted — k=%d  silhouette=%.4f", self.k, silhouette
        )
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit first.")
        return self._model.transform(df)

    def cluster_summary(self, df: DataFrame) -> DataFrame:
        """Return cluster sizes and average quality scores."""
        predictions = self.transform(df)
        return (
            predictions.groupBy("cluster_id")
            .agg(
                F.count("*").alias("doc_count"),
                F.avg("quality_score").alias("avg_quality"),
                F.avg(F.length("text_clean")).alias("avg_doc_length"),
            )
            .orderBy("cluster_id")
        )


# ---------------------------------------------------------------------------
# End-to-end Spark ML preprocessing pipeline
# ---------------------------------------------------------------------------


class SparkMLPreprocessingPipeline:
    """
    Full Spark ML preprocessing pipeline for LLM training data.

    Stages:
      1. TF-IDF feature extraction (for cluster-based sampling)
      2. Word2Vec document embeddings (dense representation)
      3. PCA reduction (for visualisation / downstream ML)
      4. K-Means clustering (corpus analysis + balanced sampling)
      5. Quality-stratified sampling (balanced cluster distribution)

    This pipeline is designed to run before fine-tuning to:
      - Remove low-quality / duplicate documents
      - Balance the training corpus across semantic clusters
      - Produce dense embeddings for retrieval pre-population
    """

    def __init__(
        self,
        n_clusters: int = 20,
        w2v_dim: int = 128,
        pca_k: int = 32,
        tfidf_features: int = 32768,
    ):
        self.tfidf = TFIDFFeaturePipeline(num_features=tfidf_features)
        self.w2v = Word2VecEmbeddingPipeline(vector_size=w2v_dim)
        self.pca = SparkPCAReducer(input_col="tfidf_features", k=pca_k)
        self.clustering = CorpusClusteringPipeline(
            feature_col="pca_features", k=n_clusters
        )

    def run(
        self,
        input_path: str,
        output_path: str,
        samples_per_cluster: int = 500,
    ) -> dict:
        """
        Full pipeline: load → TF-IDF → W2V → PCA → cluster → balanced sample → save.
        Returns a summary dict with corpus statistics.
        """
        spark = get_spark()
        df = spark.read.parquet(input_path)

        logger.info("Loaded %d documents from %s", df.count(), input_path)

        # Stage 1: TF-IDF
        df = self.tfidf.fit_transform(df)

        # Stage 2: Word2Vec
        df = self.w2v.fit_transform(df)

        # Stage 3: PCA
        df = self.pca.fit_transform(df)

        # Stage 4: Cluster
        self.clustering.fit(df)
        df = self.clustering.transform(df)

        # Stage 5: Balanced sampling — take up to N docs per cluster
        sampled = self._balanced_sample(df, samples_per_cluster)

        # Save
        (
            sampled
            .select("id", "text_clean", "source", "quality_score",
                    "cluster_id", "w2v_embedding")
            .repartition(20)
            .write.mode("overwrite")
            .parquet(output_path)
        )

        total_out = sampled.count()
        cluster_stats = self.clustering.cluster_summary(df).collect()

        summary = {
            "input_docs": df.count(),
            "output_docs": total_out,
            "n_clusters": self.clustering.k,
            "cluster_stats": [row.asDict() for row in cluster_stats],
        }
        logger.info("SparkML pipeline complete: %s", summary)
        return summary

    @staticmethod
    def _balanced_sample(df: DataFrame, n_per_cluster: int) -> DataFrame:
        """
        Reservoir-sample up to n_per_cluster documents from each cluster.
        Uses a random sort + row_number window function — fully distributed.
        """
        from pyspark.sql.window import Window

        w = Window.partitionBy("cluster_id").orderBy(F.rand(seed=42))
        return (
            df.withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") <= n_per_cluster)
            .drop("_rn")
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark ML preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--samples-per-cluster", type=int, default=500)
    args = parser.parse_args()

    pipeline = SparkMLPreprocessingPipeline(n_clusters=args.clusters)
    summary = pipeline.run(args.input, args.output, args.samples_per_cluster)
    print(f"Done: {summary}")
