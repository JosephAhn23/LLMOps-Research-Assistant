"""
Deep AWS observability - custom CloudWatch metrics, dashboards, S3 multipart,
presigned URLs, lifecycle policies.
Covers: S3 + CloudWatch (real depth)
"""
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "llmops-research-assistant")
CW_NAMESPACE = os.getenv("CW_NAMESPACE", "LLMOps/Inference")
LOG_GROUP = os.getenv("CW_LOG_GROUP", "/llmops/production")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# ─── S3 Deep Integration ──────────────────────────────────────

class S3Manager:
    """
    Production S3 integration - multipart upload, presigned URLs,
    lifecycle policies, versioning.
    """

    def __init__(self):
        self.client = boto3.client("s3", region_name=AWS_REGION)

    def setup_bucket_lifecycle(self):
        """
        Lifecycle policy: auto-tier old data, expire temp files.
        Covers: S3 lifecycle management
        """
        self.client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "transition-models-to-ia",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "models/"},
                        "Transitions": [
                            {"Days": 30, "StorageClass": "STANDARD_IA"},
                            {"Days": 90, "StorageClass": "GLACIER"},
                        ],
                    },
                    {
                        "ID": "expire-temp-data",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "tmp/"},
                        "Expiration": {"Days": 7},
                    },
                    {
                        "ID": "expire-old-versions",
                        "Status": "Enabled",
                        "Filter": {},
                        "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
                    },
                ]
            },
        )

    class ProgressCallback:
        """Track multipart upload progress."""

        def __init__(self, filename: str, total_size: int):
            self.filename = filename
            self.total_size = total_size
            self.uploaded = 0
            self._lock = threading.Lock()

        def __call__(self, bytes_transferred: int):
            with self._lock:
                self.uploaded += bytes_transferred
                pct = (self.uploaded / self.total_size) * 100
                print(
                    f"\r{self.filename}: {pct:.1f}% ({self.uploaded:,}/{self.total_size:,} bytes)",
                    end="",
                )

    def upload_model_multipart(self, local_path: str, s3_key: str):
        """
        Multipart upload for large model artifacts.
        Automatic parallelization + retry per part.
        """
        import os

        file_size = os.path.getsize(local_path)

        config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            multipart_chunksize=50 * 1024 * 1024,
            max_concurrency=10,
            use_threads=True,
        )

        callback = self.ProgressCallback(local_path, file_size)
        self.client.upload_file(
            local_path,
            S3_BUCKET,
            s3_key,
            Config=config,
            Callback=callback,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "uploaded-at": datetime.utcnow().isoformat(),
                    "model-type": "lora-finetune",
                },
            },
        )
        logger.info("Uploaded %s -> s3://%s/%s", local_path, S3_BUCKET, s3_key)

    def upload_dataframe_partitioned(
        self,
        df: pd.DataFrame,
        prefix: str,
        partition_col: str = "language",
    ):
        """
        Hive-partitioned parquet upload - mirrors production data lake patterns.
        Covers: Production data workflows
        """
        for partition_value, partition_df in df.groupby(partition_col):
            key = f"{prefix}/{partition_col}={partition_value}/data.parquet"
            buf = io.BytesIO()
            partition_df.to_parquet(buf, index=False, compression="snappy")
            buf.seek(0)
            self.client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=buf.read(),
                ContentType="application/octet-stream",
            )
            logger.info("Partition %s: %d rows -> s3://%s/%s", partition_value, len(partition_df), S3_BUCKET, key)

    def generate_presigned_url(self, s3_key: str, expiry_seconds: int = 3600) -> str:
        """Generate presigned URL for temporary model access."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=expiry_seconds,
        )

    def list_model_versions(self, prefix: str = "models/") -> List[Dict]:
        """List all model versions with metadata."""
        response = self.client.list_object_versions(Bucket=S3_BUCKET, Prefix=prefix)
        versions = []
        for v in response.get("Versions", []):
            versions.append(
                {
                    "key": v["Key"],
                    "version_id": v["VersionId"],
                    "last_modified": v["LastModified"].isoformat(),
                    "size_mb": v["Size"] / 1024 / 1024,
                    "is_latest": v["IsLatest"],
                }
            )
        return sorted(versions, key=lambda x: x["last_modified"], reverse=True)


# ─── CloudWatch Deep Integration ─────────────────────────────

@dataclass
class InferenceMetrics:
    latency_ms: float
    tokens_generated: int
    tokens_prompt: int
    retrieval_latency_ms: float
    rerank_latency_ms: float
    n_chunks_retrieved: int
    cache_hit: bool
    model_backend: str
    error: bool = False


class CloudWatchObservability:
    """
    Production CloudWatch integration - custom metrics namespace,
    structured logging, metric filters, dashboard as code.
    Covers: CloudWatch (real depth)
    """

    def __init__(self):
        self.cw = boto3.client("cloudwatch", region_name=AWS_REGION)
        self.logs = boto3.client("logs", region_name=AWS_REGION)
        self._sequence_tokens: Dict[str, str] = {}
        self._metric_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._ensure_log_group()

    def _ensure_log_group(self):
        try:
            self.logs.create_log_group(logGroupName=LOG_GROUP)
            self.logs.put_retention_policy(
                logGroupName=LOG_GROUP,
                retentionInDays=30,
            )
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass

    def emit_inference_metrics(self, metrics: InferenceMetrics):
        """
        Emit structured metrics to CloudWatch custom namespace.
        Buffers and flushes in batches for efficiency.
        """
        timestamp = datetime.utcnow()
        dimensions = [
            {"Name": "ModelBackend", "Value": metrics.model_backend},
            {"Name": "Environment", "Value": "production"},
        ]

        metric_data = [
            {
                "MetricName": "InferenceLatency",
                "Value": metrics.latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
            {
                "MetricName": "TokensGenerated",
                "Value": metrics.tokens_generated,
                "Unit": "Count",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
            {
                "MetricName": "RetrievalLatency",
                "Value": metrics.retrieval_latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
            {
                "MetricName": "CacheHitRate",
                "Value": 1.0 if metrics.cache_hit else 0.0,
                "Unit": "None",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
            {
                "MetricName": "ErrorRate",
                "Value": 1.0 if metrics.error else 0.0,
                "Unit": "None",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
            {
                "MetricName": "ChunksRetrieved",
                "Value": metrics.n_chunks_retrieved,
                "Unit": "Count",
                "Dimensions": dimensions,
                "Timestamp": timestamp,
            },
        ]

        with self._buffer_lock:
            self._metric_buffer.extend(metric_data)
            if len(self._metric_buffer) >= 20:
                self._flush_metrics()

    def _flush_metrics(self):
        """Flush buffered metrics to CloudWatch in batches of 20."""
        while self._metric_buffer:
            batch = self._metric_buffer[:20]
            self._metric_buffer = self._metric_buffer[20:]
            self.cw.put_metric_data(
                Namespace=CW_NAMESPACE,
                MetricData=batch,
            )

    def create_dashboard(self):
        """
        Create CloudWatch dashboard as code.
        Covers: Infrastructure observability as code
        """
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "title": "Inference Latency (p50/p90/p99)",
                        "metrics": [
                            [CW_NAMESPACE, "InferenceLatency", "Environment", "production",
                             {"stat": "p50", "label": "p50"}],
                            ["...", {"stat": "p90", "label": "p90"}],
                            ["...", {"stat": "p99", "label": "p99"}],
                        ],
                        "period": 60,
                        "view": "timeSeries",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Error Rate",
                        "metrics": [
                            [CW_NAMESPACE, "ErrorRate", "Environment", "production",
                             {"stat": "Average", "label": "Error Rate"}],
                        ],
                        "period": 60,
                        "view": "timeSeries",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Cache Hit Rate",
                        "metrics": [
                            [CW_NAMESPACE, "CacheHitRate", "Environment", "production",
                             {"stat": "Average"}],
                        ],
                        "period": 300,
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Tokens Generated",
                        "metrics": [
                            [CW_NAMESPACE, "TokensGenerated", "Environment", "production",
                             {"stat": "Sum", "label": "Total tokens/min"}],
                        ],
                        "period": 60,
                    },
                },
            ]
        }
        self.cw.put_dashboard(
            DashboardName="LLMOps-Production",
            DashboardBody=json.dumps(dashboard_body),
        )
        logger.info("CloudWatch dashboard created: LLMOps-Production")

    def create_alarms(self):
        """Create CloudWatch alarms for SLA monitoring."""
        alarms = [
            {
                "AlarmName": "LLMOps-HighLatency-p99",
                "MetricName": "InferenceLatency",
                "ExtendedStatistic": "p99",
                "Threshold": 5000,
                "ComparisonOperator": "GreaterThanThreshold",
                "AlarmDescription": "p99 inference latency > 5s",
            },
            {
                "AlarmName": "LLMOps-HighErrorRate",
                "MetricName": "ErrorRate",
                "Statistic": "Average",
                "Threshold": 0.05,
                "ComparisonOperator": "GreaterThanThreshold",
                "AlarmDescription": "Error rate > 5%",
            },
            {
                "AlarmName": "LLMOps-LowCacheHitRate",
                "MetricName": "CacheHitRate",
                "Statistic": "Average",
                "Threshold": 0.20,
                "ComparisonOperator": "LessThanThreshold",
                "AlarmDescription": "Cache hit rate < 20%",
            },
        ]

        for alarm in alarms:
            kwargs = {
                "AlarmName": alarm["AlarmName"],
                "MetricName": alarm["MetricName"],
                "Namespace": CW_NAMESPACE,
                "Dimensions": [{"Name": "Environment", "Value": "production"}],
                "Period": 60,
                "EvaluationPeriods": 3,
                "Threshold": alarm["Threshold"],
                "ComparisonOperator": alarm["ComparisonOperator"],
                "AlarmDescription": alarm["AlarmDescription"],
                "TreatMissingData": "notBreaching",
            }
            if "ExtendedStatistic" in alarm:
                kwargs["ExtendedStatistic"] = alarm["ExtendedStatistic"]
            else:
                kwargs["Statistic"] = alarm.get("Statistic", "Average")
            self.cw.put_metric_alarm(**kwargs)

        logger.info("Created %d CloudWatch alarms", len(alarms))

    def create_metric_filter(self):
        """Extract metrics from structured logs automatically."""
        self.logs.put_metric_filter(
            logGroupName=LOG_GROUP,
            filterName="InferenceErrors",
            filterPattern='{ $.event = "inference_error" }',
            metricTransformations=[{
                "metricName": "InferenceErrors",
                "metricNamespace": CW_NAMESPACE,
                "metricValue": "1",
                "defaultValue": 0,
            }],
        )

    def log_structured(self, log_stream: str, event: Dict[str, Any]):
        """Structured JSON logging to CloudWatch Logs."""
        try:
            self.logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=log_stream)
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass

        kwargs = {
            "logGroupName": LOG_GROUP,
            "logStreamName": log_stream,
            "logEvents": [{
                "timestamp": int(time.time() * 1000),
                "message": json.dumps({**event, "timestamp": datetime.utcnow().isoformat()}),
            }],
        }
        if log_stream in self._sequence_tokens:
            kwargs["sequenceToken"] = self._sequence_tokens[log_stream]

        response = self.logs.put_log_events(**kwargs)
        self._sequence_tokens[log_stream] = response.get("nextSequenceToken", "")
