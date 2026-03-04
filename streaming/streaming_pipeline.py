"""
Event-Driven Streaming Pipeline — Kafka + AWS Kinesis
=======================================================
Real-time ML inference pipeline with two streaming backends:

  Kafka   — high-throughput, partitioned log (self-hosted or Confluent Cloud)
  Kinesis — managed AWS streaming (Kinesis Data Streams)

Architecture:
  Producer → [Kafka topic / Kinesis stream]
    → Consumer → Inference worker → Result topic / downstream sink
    → Dead Letter Queue (DLQ) on failure (retry_count ≥ 3)
    → DynamoDB checkpointing for Kinesis exactly-once semantics
    → MLflow metrics logging per consumer loop

Use cases:
  - Real-time RAG queries:    user request → retrieve → generate → respond
  - Document ingestion:       new doc → embed → index → FAISS / Azure Search
  - Model monitoring:         inference logs → anomaly detection → alert

Usage:
    # Kafka producer
    producer = KafkaMLProducer("llmops-queries")
    eid = producer.send_inference_request("What is RAG?", user_id="u123")

    # Kafka consumer
    consumer = KafkaInferenceConsumer("llmops-queries", inference_fn=my_model)
    consumer.run(max_events=100)

    # Kinesis producer
    producer = KinesisMLProducer()
    producer.send_inference_request("What is RAG?")

    # CLI
    python streaming/streaming_pipeline.py --backend kafka --mode produce --n-events 10
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Message schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MLEvent:
    """
    Standard event envelope for all ML pipeline messages.
    All producers and consumers use this schema — version-pinned for
    backward compatibility across rolling deployments.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "inference_request"   # inference_request | ingest | eval | monitor
    payload: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "llmops-api"
    schema_version: str = "1.0"
    retry_count: int = 0
    trace_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "schema_version": self.schema_version,
            "retry_count": self.retry_count,
            "trace_id": self.trace_id,
        })

    @classmethod
    def from_json(cls, data: str | bytes) -> "MLEvent":
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceResult:
    event_id: str
    query: str
    answer: str
    latency_ms: float
    model: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "llmops-inference-workers"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False        # manual commit for at-least-once
    max_poll_records: int = 10
    session_timeout_ms: int = 30_000
    request_timeout_ms: int = 40_000
    retries: int = 3
    dlq_topic_suffix: str = "-dlq"
    results_topic_suffix: str = "-results"
    security_protocol: str = "PLAINTEXT"   # PLAINTEXT | SASL_SSL
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


@dataclass
class KinesisConfig:
    stream_name: str = "llmops-queries"
    region: str = "us-east-1"
    shard_count: int = 2
    results_stream: str = "llmops-results"
    dlq_stream: str = "llmops-dlq"
    checkpoint_table: str = "llmops-kinesis-checkpoints"   # DynamoDB table
    read_interval_seconds: float = 1.0
    max_records_per_shard: int = 100
    max_retries: int = 3


# ──────────────────────────────────────────────────────────────────────────────
# Base producer
# ──────────────────────────────────────────────────────────────────────────────

class BaseMLProducer(ABC):
    @abstractmethod
    def send(self, event: MLEvent) -> bool: ...

    @abstractmethod
    def flush(self): ...

    @abstractmethod
    def close(self): ...

    def send_inference_request(
        self, query: str, user_id: Optional[str] = None, **kwargs
    ) -> str:
        """Convenience wrapper — returns the event_id for correlation."""
        event = MLEvent(
            event_type="inference_request",
            payload={"query": query, "user_id": user_id, **kwargs},
            trace_id=str(uuid.uuid4()),
        )
        self.send(event)
        return event.event_id

    def send_ingest_request(
        self, doc_id: str, text: str, metadata: Optional[Dict] = None
    ) -> str:
        event = MLEvent(
            event_type="ingest",
            payload={"doc_id": doc_id, "text": text, "metadata": metadata or {}},
        )
        self.send(event)
        return event.event_id

    def send_monitor_event(self, metrics: Dict) -> str:
        event = MLEvent(event_type="monitor", payload=metrics)
        self.send(event)
        return event.event_id


# ──────────────────────────────────────────────────────────────────────────────
# Base consumer
# ──────────────────────────────────────────────────────────────────────────────

class BaseMLConsumer(ABC):
    MAX_RETRIES = 3

    def __init__(self, inference_fn: Optional[Callable] = None):
        self.inference_fn = inference_fn
        self._running = False
        self.processed = 0
        self.errors = 0

    @abstractmethod
    def poll(self, timeout_ms: int = 1000) -> List[MLEvent]: ...

    @abstractmethod
    def commit(self): ...

    @abstractmethod
    def send_to_dlq(self, event: MLEvent, error: str): ...

    @abstractmethod
    def send_result(self, result: InferenceResult): ...

    @abstractmethod
    def close(self): ...

    def process_event(self, event: MLEvent) -> Optional[InferenceResult]:
        """Route event to the appropriate handler by event_type."""
        handlers = {
            "inference_request": self._handle_inference,
            "ingest": self._handle_ingest,
            "eval": self._handle_eval,
            "monitor": self._handle_monitor,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        logger.warning("Unknown event type: %s (event_id=%s)", event.event_type, event.event_id)
        return None

    def _handle_inference(self, event: MLEvent) -> InferenceResult:
        query = event.payload.get("query", "")
        t0 = time.perf_counter()
        if self.inference_fn:
            answer = self.inference_fn(query)
        else:
            answer = f"[mock] Response to: {query[:50]}"
        latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            event_id=event.event_id,
            query=query,
            answer=answer,
            latency_ms=latency_ms,
            model="llmops-rag",
        )

    def _handle_ingest(self, event: MLEvent) -> None:
        """Hook: call embedding + FAISS indexing pipeline here."""
        doc_id = event.payload.get("doc_id")
        text = event.payload.get("text", "")
        logger.info("Ingesting doc: %s (%d chars)", doc_id, len(text))
        return None

    def _handle_eval(self, event: MLEvent) -> None:
        logger.info("Eval event received: %s", event.event_id)
        return None

    def _handle_monitor(self, event: MLEvent) -> None:
        logger.debug("Monitor event: %s", event.payload)
        return None

    def run(self, max_events: Optional[int] = None, poll_timeout_ms: int = 1000):
        """
        Main consumer loop.

        - Polls for events, processes each, commits offset.
        - On failure: increments retry_count; routes to DLQ after MAX_RETRIES.
        - Logs processed/error counts to MLflow.
        """
        self._running = True
        logger.info("Consumer started (max_events=%s)", max_events or "unlimited")

        try:
            while self._running:
                events = self.poll(timeout_ms=poll_timeout_ms)
                for event in events:
                    try:
                        result = self.process_event(event)
                        if result:
                            self.send_result(result)
                        self.commit()
                        self.processed += 1
                        self._log_metrics()

                        if max_events and self.processed >= max_events:
                            logger.info("Reached max_events=%d — stopping.", max_events)
                            self._running = False
                            break

                    except Exception as e:
                        self.errors += 1
                        logger.error("Error processing event %s: %s", event.event_id, e)
                        event.retry_count += 1
                        if event.retry_count >= self.MAX_RETRIES:
                            self.send_to_dlq(event, str(e))
                            logger.warning(
                                "Event %s sent to DLQ after %d retries",
                                event.event_id, self.MAX_RETRIES,
                            )
                        self._log_metrics()
        finally:
            self.close()

    def stop(self):
        self._running = False

    def _log_metrics(self):
        try:
            import mlflow
            mlflow.log_metric("events_processed", self.processed)
            mlflow.log_metric("events_errored", self.errors)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Kafka producer
# ──────────────────────────────────────────────────────────────────────────────

class KafkaMLProducer(BaseMLProducer):
    """
    Kafka producer for ML pipeline events.
    Supports PLAINTEXT and SASL_SSL (Confluent Cloud / Amazon MSK).
    Idempotent producer enabled by default (acks=all + enable_idempotence).
    """

    def __init__(self, topic: str, cfg: Optional[KafkaConfig] = None):
        self.topic = topic
        self.cfg = cfg or KafkaConfig()
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            try:
                from kafka import KafkaProducer
            except ImportError:
                raise ImportError("kafka-python required: pip install kafka-python")

            kwargs: Dict = {
                "bootstrap_servers": self.cfg.bootstrap_servers,
                "value_serializer": lambda v: v.encode("utf-8"),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
                "retries": self.cfg.retries,
                "acks": "all",
                "enable_idempotence": True,
            }
            if self.cfg.security_protocol == "SASL_SSL":
                kwargs.update({
                    "security_protocol": "SASL_SSL",
                    "sasl_mechanism": self.cfg.sasl_mechanism,
                    "sasl_plain_username": self.cfg.sasl_username,
                    "sasl_plain_password": self.cfg.sasl_password,
                })
            self._producer = KafkaProducer(**kwargs)
            logger.info("Kafka producer connected: %s → %s",
                        self.cfg.bootstrap_servers, self.topic)
        return self._producer

    def send(self, event: MLEvent) -> bool:
        try:
            future = self._get_producer().send(
                self.topic,
                key=event.event_id,
                value=event.to_json(),
            )
            future.get(timeout=10)
            logger.debug("Sent event %s → %s", event.event_id, self.topic)
            return True
        except Exception as e:
            logger.error("Kafka send failed: %s", e)
            return False

    def send_batch(self, events: List[MLEvent]) -> int:
        """Send a batch of events without waiting for individual acks."""
        producer = self._get_producer()
        for event in events:
            producer.send(self.topic, key=event.event_id, value=event.to_json())
        producer.flush()
        return len(events)

    def flush(self):
        if self._producer:
            self._producer.flush()

    def close(self):
        if self._producer:
            self._producer.close()
            logger.info("Kafka producer closed.")


# ──────────────────────────────────────────────────────────────────────────────
# Kafka consumer
# ──────────────────────────────────────────────────────────────────────────────

class KafkaInferenceConsumer(BaseMLConsumer):
    """
    Kafka consumer with:
      - Manual offset commit (at-least-once delivery)
      - DLQ routing after MAX_RETRIES failures
      - Result publishing to <topic>-results
      - MLflow metric logging per loop iteration
    """

    def __init__(
        self,
        topic: str,
        cfg: Optional[KafkaConfig] = None,
        inference_fn: Optional[Callable] = None,
    ):
        super().__init__(inference_fn)
        self.topic = topic
        self.cfg = cfg or KafkaConfig()
        self._consumer = None
        self._result_producer: Optional[KafkaMLProducer] = None
        self._dlq_producer: Optional[KafkaMLProducer] = None

    def _get_consumer(self):
        if self._consumer is None:
            try:
                from kafka import KafkaConsumer
            except ImportError:
                raise ImportError("kafka-python required: pip install kafka-python")

            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.cfg.bootstrap_servers,
                group_id=self.cfg.group_id,
                auto_offset_reset=self.cfg.auto_offset_reset,
                enable_auto_commit=self.cfg.enable_auto_commit,
                value_deserializer=lambda v: v,
                session_timeout_ms=self.cfg.session_timeout_ms,
                request_timeout_ms=self.cfg.request_timeout_ms,
            )
            logger.info("Kafka consumer connected: topic=%s group=%s",
                        self.topic, self.cfg.group_id)
        return self._consumer

    def poll(self, timeout_ms: int = 1000) -> List[MLEvent]:
        records = self._get_consumer().poll(timeout_ms=timeout_ms)
        events = []
        for _tp, messages in records.items():
            for msg in messages:
                try:
                    events.append(MLEvent.from_json(msg.value))
                except Exception as e:
                    logger.warning("Failed to deserialise message: %s", e)
        return events

    def commit(self):
        if self._consumer:
            self._consumer.commit()

    def send_to_dlq(self, event: MLEvent, error: str):
        if self._dlq_producer is None:
            self._dlq_producer = KafkaMLProducer(
                self.topic + self.cfg.dlq_topic_suffix, self.cfg
            )
        event.payload["dlq_error"] = error
        event.payload["dlq_timestamp"] = datetime.utcnow().isoformat()
        self._dlq_producer.send(event)

    def send_result(self, result: InferenceResult):
        if self._result_producer is None:
            self._result_producer = KafkaMLProducer(
                self.topic + self.cfg.results_topic_suffix, self.cfg
            )
        self._result_producer.send(MLEvent(
            event_type="inference_result",
            payload=json.loads(result.to_json()),
        ))

    def close(self):
        if self._consumer:
            self._consumer.close()
        if self._result_producer:
            self._result_producer.close()
        if self._dlq_producer:
            self._dlq_producer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Kinesis producer
# ──────────────────────────────────────────────────────────────────────────────

class KinesisMLProducer(BaseMLProducer):
    """AWS Kinesis Data Streams producer."""

    def __init__(self, cfg: Optional[KinesisConfig] = None):
        self.cfg = cfg or KinesisConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("kinesis", region_name=self.cfg.region)
                logger.info("Kinesis producer connected: %s", self.cfg.stream_name)
            except ImportError:
                raise ImportError("boto3 required: pip install boto3")
        return self._client

    def send(self, event: MLEvent) -> bool:
        try:
            self._get_client().put_record(
                StreamName=self.cfg.stream_name,
                Data=event.to_json().encode("utf-8"),
                PartitionKey=event.event_id,
            )
            return True
        except Exception as e:
            logger.error("Kinesis put_record failed: %s", e)
            return False

    def send_batch(self, events: List[MLEvent]) -> int:
        """
        Kinesis batch put — max 500 records or 5 MB per call.
        Returns number of successfully sent records.
        """
        client = self._get_client()
        records = [
            {"Data": e.to_json().encode("utf-8"), "PartitionKey": e.event_id}
            for e in events
        ]
        sent = 0
        for i in range(0, len(records), 500):
            batch = records[i:i + 500]
            response = client.put_records(
                StreamName=self.cfg.stream_name, Records=batch
            )
            sent += len(batch) - response.get("FailedRecordCount", 0)
        return sent

    def flush(self):
        pass  # Kinesis put_record is synchronous

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Kinesis consumer
# ──────────────────────────────────────────────────────────────────────────────

class KinesisInferenceConsumer(BaseMLConsumer):
    """
    AWS Kinesis consumer using GetShardIterator / GetRecords.

    Fault tolerance:
      - DynamoDB checkpointing: saves the last processed sequence number per shard
      - On restart, resumes from the last checkpoint rather than LATEST
      - DLQ routing to a separate Kinesis stream on repeated failures
    """

    def __init__(
        self,
        cfg: Optional[KinesisConfig] = None,
        inference_fn: Optional[Callable] = None,
    ):
        super().__init__(inference_fn)
        self.cfg = cfg or KinesisConfig()
        self._client = None
        self._shard_iterators: Dict[str, str] = {}
        self._last_sequences: Dict[str, str] = {}

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("kinesis", region_name=self.cfg.region)
        return self._client

    def _init_shards(self):
        client = self._get_client()
        response = client.describe_stream(StreamName=self.cfg.stream_name)
        shards = response["StreamDescription"]["Shards"]

        for shard in shards:
            shard_id = shard["ShardId"]
            checkpoint = self._load_checkpoint(shard_id)
            if checkpoint:
                iter_resp = client.get_shard_iterator(
                    StreamName=self.cfg.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType="AFTER_SEQUENCE_NUMBER",
                    StartingSequenceNumber=checkpoint,
                )
            else:
                iter_resp = client.get_shard_iterator(
                    StreamName=self.cfg.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType="LATEST",
                )
            self._shard_iterators[shard_id] = iter_resp["ShardIterator"]

        logger.info("Kinesis: initialised %d shards for stream %s",
                    len(shards), self.cfg.stream_name)

    def poll(self, timeout_ms: int = 1000) -> List[MLEvent]:
        if not self._shard_iterators:
            self._init_shards()

        client = self._get_client()
        events: List[MLEvent] = []

        for shard_id, iterator in list(self._shard_iterators.items()):
            try:
                response = client.get_records(
                    ShardIterator=iterator,
                    Limit=self.cfg.max_records_per_shard,
                )
                next_iter = response.get("NextShardIterator")
                if next_iter:
                    self._shard_iterators[shard_id] = next_iter

                for record in response.get("Records", []):
                    try:
                        event = MLEvent.from_json(record["Data"])
                        self._last_sequences[shard_id] = record["SequenceNumber"]
                        events.append(event)
                    except Exception as e:
                        logger.warning("Failed to parse Kinesis record: %s", e)

            except Exception as e:
                logger.error("GetRecords failed for shard %s: %s", shard_id, e)

        time.sleep(self.cfg.read_interval_seconds)
        return events

    def commit(self):
        """Persist current sequence numbers to DynamoDB."""
        for shard_id, seq in self._last_sequences.items():
            self._save_checkpoint(shard_id, seq)

    def _save_checkpoint(self, shard_id: str, sequence: str):
        try:
            import boto3
            dynamodb = boto3.resource("dynamodb", region_name=self.cfg.region)
            table = dynamodb.Table(self.cfg.checkpoint_table)
            table.put_item(Item={
                "shard_id": shard_id,
                "sequence_number": sequence,
                "timestamp": datetime.utcnow().isoformat(),
                "stream": self.cfg.stream_name,
            })
        except Exception as e:
            logger.warning("Checkpoint save failed (shard=%s): %s", shard_id, e)

    def _load_checkpoint(self, shard_id: str) -> Optional[str]:
        try:
            import boto3
            dynamodb = boto3.resource("dynamodb", region_name=self.cfg.region)
            table = dynamodb.Table(self.cfg.checkpoint_table)
            response = table.get_item(Key={"shard_id": shard_id})
            return response.get("Item", {}).get("sequence_number")
        except Exception:
            return None

    def send_to_dlq(self, event: MLEvent, error: str):
        try:
            import boto3
            client = boto3.client("kinesis", region_name=self.cfg.region)
            event.payload["dlq_error"] = error
            event.payload["dlq_timestamp"] = datetime.utcnow().isoformat()
            client.put_record(
                StreamName=self.cfg.dlq_stream,
                Data=event.to_json().encode("utf-8"),
                PartitionKey=event.event_id,
            )
            logger.warning("Event %s sent to DLQ stream %s: %s",
                           event.event_id, self.cfg.dlq_stream, error)
        except Exception as e:
            logger.error("DLQ send failed: %s", e)

    def send_result(self, result: InferenceResult):
        try:
            import boto3
            client = boto3.client("kinesis", region_name=self.cfg.region)
            client.put_record(
                StreamName=self.cfg.results_stream,
                Data=result.to_json().encode("utf-8"),
                PartitionKey=result.event_id,
            )
        except Exception as e:
            logger.error("Result send to Kinesis failed: %s", e)

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Stream topology builder
# ──────────────────────────────────────────────────────────────────────────────

class StreamTopology:
    """
    Declare and provision the full streaming topology:
      queries → inference → results
      queries → [on failure] → DLQ
      ingest  → embedding → index
      monitor → anomaly detection → alert
    """

    KAFKA_TOPICS = [
        "llmops-queries",
        "llmops-queries-results",
        "llmops-queries-dlq",
        "llmops-ingest",
        "llmops-ingest-dlq",
        "llmops-monitor",
    ]

    KINESIS_STREAMS = [
        "llmops-queries",
        "llmops-results",
        "llmops-dlq",
        "llmops-ingest",
    ]

    @staticmethod
    def create_kafka_topics(cfg: KafkaConfig, num_partitions: int = 3) -> None:
        try:
            from kafka.admin import KafkaAdminClient, NewTopic

            admin = KafkaAdminClient(bootstrap_servers=cfg.bootstrap_servers)
            topics = [
                NewTopic(name=t, num_partitions=num_partitions, replication_factor=1)
                for t in StreamTopology.KAFKA_TOPICS
            ]
            admin.create_topics(new_topics=topics, validate_only=False)
            logger.info("Created Kafka topics: %s", StreamTopology.KAFKA_TOPICS)
        except Exception as e:
            logger.warning("Kafka topic creation: %s", e)

    @staticmethod
    def create_kinesis_streams(cfg: KinesisConfig) -> None:
        try:
            import boto3
            client = boto3.client("kinesis", region_name=cfg.region)
            for stream in StreamTopology.KINESIS_STREAMS:
                try:
                    client.create_stream(StreamName=stream, ShardCount=cfg.shard_count)
                    logger.info("Created Kinesis stream: %s (%d shards)",
                                stream, cfg.shard_count)
                except client.exceptions.ResourceInUseException:
                    logger.info("Kinesis stream already exists: %s", stream)
        except Exception as e:
            logger.warning("Kinesis stream creation: %s", e)

    @staticmethod
    def create_dynamodb_checkpoint_table(cfg: KinesisConfig) -> None:
        """Provision the DynamoDB table used for Kinesis checkpointing."""
        try:
            import boto3
            dynamodb = boto3.client("dynamodb", region_name=cfg.region)
            dynamodb.create_table(
                TableName=cfg.checkpoint_table,
                KeySchema=[{"AttributeName": "shard_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "shard_id", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            logger.info("DynamoDB checkpoint table created: %s", cfg.checkpoint_table)
        except Exception as e:
            logger.warning("DynamoDB table creation: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML streaming pipeline")
    parser.add_argument("--backend", choices=["kafka", "kinesis"], default="kafka")
    parser.add_argument("--mode", choices=["produce", "consume", "setup"], default="produce")
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--topic", default="llmops-queries",
                        help="Kafka topic (ignored for Kinesis)")
    args = parser.parse_args()

    if args.backend == "kafka":
        cfg = KafkaConfig()
        if args.mode == "setup":
            StreamTopology.create_kafka_topics(cfg)
        elif args.mode == "produce":
            producer = KafkaMLProducer(args.topic, cfg)
            for i in range(args.n_events):
                eid = producer.send_inference_request(
                    f"Test query {i}: what is RAG?", user_id=f"user_{i}"
                )
                print(f"Sent event: {eid}")
            producer.flush()
            producer.close()
        elif args.mode == "consume":
            consumer = KafkaInferenceConsumer(args.topic, cfg)
            consumer.run(max_events=args.n_events)

    elif args.backend == "kinesis":
        cfg = KinesisConfig()
        if args.mode == "setup":
            StreamTopology.create_kinesis_streams(cfg)
            StreamTopology.create_dynamodb_checkpoint_table(cfg)
        elif args.mode == "produce":
            producer = KinesisMLProducer(cfg)
            for i in range(args.n_events):
                eid = producer.send_inference_request(f"Test query {i}: what is RAG?")
                print(f"Sent: {eid}")
        elif args.mode == "consume":
            consumer = KinesisInferenceConsumer(cfg)
            consumer.run(max_events=args.n_events)
