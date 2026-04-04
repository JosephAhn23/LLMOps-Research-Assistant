"""
AWS CloudWatch logging + S3 data storage integration.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOG_GROUP = os.getenv("CW_LOG_GROUP", "/llmops-research-assistant")
LOG_STREAM = f"inference-{datetime.now().strftime('%Y%m%d')}"
S3_BUCKET = os.getenv("S3_BUCKET", "your-llmops-bucket")


class CloudWatchLogger:
    """
    Structured logging to AWS CloudWatch.
    Covers: CloudWatch monitoring
    """

    def __init__(self):
        self.client = boto3.client("logs", region_name=AWS_REGION)
        self.metrics_client = boto3.client("cloudwatch", region_name=AWS_REGION)
        self._ensure_log_group()
        self._ensure_log_stream()
        self.sequence_token = None

    def _ensure_log_group(self):
        try:
            self.client.create_log_group(logGroupName=LOG_GROUP)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def _ensure_log_stream(self):
        try:
            self.client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def log(self, event: Dict[str, Any]):
        kwargs = dict(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[
                {
                    "timestamp": int(time.time() * 1000),
                    "message": json.dumps(event),
                }
            ],
        )
        if self.sequence_token:
            kwargs["sequenceToken"] = self.sequence_token

        response = self.client.put_log_events(**kwargs)
        self.sequence_token = response.get("nextSequenceToken")

    def log_inference(self, query: str, latency: float, tokens: int, sources: list):
        self.log(
            {
                "event": "inference",
                "query_length": len(query),
                "latency_ms": round(latency * 1000, 2),
                "tokens_used": tokens,
                "n_sources": len(sources),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def emit_metric(self, name: str, value: float, unit: str = "Count", model_name: str = "unknown"):
        self.metrics_client.put_metric_data(
            Namespace="LLMOps/Inference",
            MetricData=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": [
                        {"Name": "Environment", "Value": os.getenv("ENVIRONMENT", "production")},
                        {"Name": "Model", "Value": model_name},
                    ],
                }
            ],
        )


class S3DataStore:
    """
    S3 integration for dataset storage + model artifacts.
    Covers: S3, cloud data storage
    """

    def __init__(self):
        self.client = boto3.client("s3", region_name=AWS_REGION)

    def upload_dataframe(self, df: pd.DataFrame, key: str):
        """Upload processed DataFrame as parquet to S3."""
        import io

        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
        logger.info("Uploaded %d rows to s3://%s/%s", len(df), S3_BUCKET, key)

    def download_dataframe(self, key: str) -> pd.DataFrame:
        import io

        response = self.client.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(response["Body"].read()))

    def upload_model_artifact(self, local_path: str, s3_key: str):
        self.client.upload_file(local_path, S3_BUCKET, s3_key)
        logger.info("Uploaded model to s3://%s/%s", S3_BUCKET, s3_key)

    def upload_large_model(self, local_path: str, s3_key: str, chunk_size_mb: int = 100):
        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=chunk_size_mb * 1024 * 1024,
            multipart_chunksize=chunk_size_mb * 1024 * 1024,
            max_concurrency=10,
            use_threads=True,
        )
        self.client.upload_file(local_path, S3_BUCKET, s3_key, Config=transfer_config)
        logger.info("Multipart upload complete: s3://%s/%s", S3_BUCKET, s3_key)

    def create_presigned_download_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_in_seconds,
        )

    def apply_lifecycle_policy(self, prefix: str = "artifacts/", expiration_days: int = 30):
        self.client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "llmops-artifact-expiry",
                        "Filter": {"Prefix": prefix},
                        "Status": "Enabled",
                        "Expiration": {"Days": expiration_days},
                    }
                ]
            },
        )
