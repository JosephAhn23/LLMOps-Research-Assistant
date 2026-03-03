"""
Apache Spark pipeline for large-scale distributed data processing.
Covers: Apache Spark, distributed data processing, scalable pipelines
"""
import re
from typing import Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("LLMOpsIngestion")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def clean_text_udf(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_quality_score(text: str) -> float:
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 10:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    score = min(avg_len / 8.0, 1.0) * (1.0 - digit_ratio)
    return float(score)


class SparkIngestionPipeline:
    """
    Distributed text processing pipeline using Apache Spark.
    Processes millions of documents across partitions.
    """

    def __init__(self):
        self.spark = get_spark()
        self.clean_udf = F.udf(clean_text_udf, StringType())
        self.quality_udf = F.udf(compute_quality_score, FloatType())

    def process_jsonl(self, input_path: str, output_path: str, min_quality: float = 0.4):
        """
        Full pipeline: load -> clean -> score -> filter -> deduplicate -> save.
        """
        df = self.spark.read.json(input_path)
        df = df.withColumn("text_clean", self.clean_udf(F.col("text")))
        df = df.withColumn("quality_score", self.quality_udf(F.col("text_clean")))
        df = df.filter(F.col("quality_score") >= min_quality)
        df = df.withColumn("content_hash", F.md5(F.col("text_clean")))
        df = df.dropDuplicates(["content_hash"])

        total = df.count()
        print(f"Processed {total} quality documents")

        (
            df.select("id", "text_clean", "source", "quality_score")
            .repartition(50)
            .write.mode("overwrite")
            .parquet(output_path)
        )

        return df

    def compute_dataset_statistics(self, df) -> Dict:
        """High-dimensional data analysis over corpus."""
        stats = (
            df.agg(
                F.count("*").alias("total_docs"),
                F.avg("quality_score").alias("avg_quality"),
                F.stddev("quality_score").alias("std_quality"),
                F.avg(F.length("text_clean")).alias("avg_doc_length"),
                F.countDistinct("source").alias("unique_sources"),
            )
            .collect()[0]
            .asDict()
        )
        return stats
