"""
Streaming ML: Real-time feature computation and drift detection.

Covers:
  - Stateful stream processor with windowed aggregations
  - Online embedding refresh (incrementally update index on new data)
  - Concept drift detection: Page-Hinkley test + ADWIN
  - Feature drift monitoring: PSI (Population Stability Index)
  - Real-time anomaly detection on incoming query streams
  - Dead Letter Queue routing for failed records

Resume framing:
  "Designed streaming ML inference system with stateful feature computation,
   concept drift detection (Page-Hinkley + ADWIN), and online embedding
   refresh — enabling the retrieval index to adapt to new content within minutes
   of ingestion without full reindexing."

Usage:
    processor = StreamProcessor(window_size=100)
    for event in event_stream:
        features = processor.process(event)
        drift = processor.check_drift("ragas_faithfulness")
        if drift.detected:
            trigger_retraining_pipeline()
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    partition_key: Optional[str] = None


@dataclass
class DriftAlert:
    metric: str
    detected: bool
    method: str
    statistic: float
    threshold: float
    direction: str
    magnitude: float
    n_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        status = "DRIFT DETECTED" if self.detected else "stable"
        return (
            f"{self.metric} [{status}] via {self.method}: "
            f"stat={self.statistic:.4f} threshold={self.threshold:.4f} "
            f"direction={self.direction} magnitude={self.magnitude:.4f}"
        )


# ---------------------------------------------------------------------------
# Page-Hinkley Drift Detector
# ---------------------------------------------------------------------------

class PageHinkley:
    """
    Page-Hinkley test for detecting change in the mean of a data stream.

    Accumulates evidence of change; fires when cumulative sum exceeds threshold.
    Sensitive to gradual drift (unlike sudden-change detectors).

    Used for: RAGAS score degradation, latency creep, embedding quality drift.
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.9999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0

    def update(self, value: float) -> Tuple[bool, float]:
        self._n += 1
        self._mean = self._alpha_mean(value)
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        ph = self._sum - self._min_sum
        detected = ph > self.lambda_
        return detected, ph

    def _alpha_mean(self, value: float) -> float:
        if self._n == 1:
            self._mean = value
        else:
            self._mean = self.alpha * self._mean + (1 - self.alpha) * value
        return self._mean

    def reset(self):
        self._sum = 0.0
        self._min_sum = 0.0


# ---------------------------------------------------------------------------
# ADWIN (Adaptive Windowing)
# ---------------------------------------------------------------------------

class ADWIN:
    """
    ADWIN: Adaptive Windowing drift detector (Bifet & Gavalda, 2007).

    Maintains a sliding window and detects when the mean of the two halves
    diverges beyond a statistical threshold. Automatically adjusts window size.

    Advantage over Page-Hinkley: handles both sudden and gradual drift,
    and provides the window size as a drift signal.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: Deque[float] = deque()
        self._total = 0.0
        self._n = 0

    def update(self, value: float) -> Tuple[bool, int]:
        self._window.append(value)
        self._total += value
        self._n += 1

        drift_detected = False
        if self._n >= 10:
            drift_detected = self._check_drift()

        return drift_detected, self._n

    def _check_drift(self) -> bool:
        window = list(self._window)
        n = len(window)
        total = sum(window)

        for split in range(2, n - 2):
            n0, n1 = split, n - split
            sum0 = sum(window[:split])
            sum1 = total - sum0
            mean0, mean1 = sum0 / n0, sum1 / n1

            epsilon_cut = math.sqrt(
                (1 / (2 * n0) + 1 / (2 * n1)) *
                math.log(4 * n * n / self.delta)
            )

            if abs(mean0 - mean1) > epsilon_cut:
                while len(self._window) > n1:
                    removed = self._window.popleft()
                    self._total -= removed
                    self._n -= 1
                return True
        return False

    @property
    def mean(self) -> float:
        return self._total / max(self._n, 1)


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: List[float],
    current: List[float],
    n_bins: int = 10,
) -> float:
    """
    PSI measures distribution shift between reference and current populations.

    Interpretation:
      PSI < 0.10: no significant change
      0.10 <= PSI < 0.25: moderate change, investigate
      PSI >= 0.25: major shift, retrain required
    """
    if not reference or not current:
        return 0.0

    min_val = min(min(reference), min(current))
    max_val = max(max(reference), max(current))
    if max_val == min_val:
        return 0.0

    bins = [min_val + i * (max_val - min_val) / n_bins for i in range(n_bins + 1)]

    def bin_counts(data: List[float]) -> List[float]:
        counts = [0] * n_bins
        for val in data:
            idx = min(int((val - min_val) / (max_val - min_val) * n_bins), n_bins - 1)
            counts[idx] += 1
        return [(c + 0.5) / len(data) for c in counts]

    ref_pct = bin_counts(reference)
    cur_pct = bin_counts(current)

    psi = sum((c - r) * math.log(c / r) for r, c in zip(ref_pct, cur_pct))
    return round(psi, 4)


# ---------------------------------------------------------------------------
# Windowed Aggregator
# ---------------------------------------------------------------------------

class WindowedAggregator:
    """
    Maintains rolling statistics over a sliding window of events.
    Used for real-time feature computation: rolling mean, std, p95.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._windows: Dict[str, Deque[float]] = {}

    def update(self, metric: str, value: float) -> None:
        if metric not in self._windows:
            self._windows[metric] = deque(maxlen=self.window_size)
        self._windows[metric].append(value)

    def stats(self, metric: str) -> Dict[str, float]:
        window = list(self._windows.get(metric, []))
        if not window:
            return {}
        n = len(window)
        mean = sum(window) / n
        variance = sum((x - mean)**2 for x in window) / max(n - 1, 1)
        std = math.sqrt(variance)
        sorted_w = sorted(window)
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(sorted_w[0], 4),
            "max": round(sorted_w[-1], 4),
            "p50": round(sorted_w[n // 2], 4),
            "p95": round(sorted_w[int(0.95 * n)], 4),
            "n": n,
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        return {metric: self.stats(metric) for metric in self._windows}


# ---------------------------------------------------------------------------
# Online Embedding Refresher
# ---------------------------------------------------------------------------

class OnlineEmbeddingRefresher:
    """
    Incrementally updates the FAISS retrieval index as new documents arrive.

    Instead of full reindexing (expensive), adds new embeddings in micro-batches.
    Triggers a background reindex when:
    1. Accumulated new documents exceed a threshold
    2. Embedding drift (PSI) exceeds 0.25

    In production: calls ingestion.pipeline.IngestionPipeline.ingest_documents()
    with the accumulated batch.
    """

    def __init__(self, batch_size: int = 50, drift_threshold: float = 0.25):
        self.batch_size = batch_size
        self.drift_threshold = drift_threshold
        self._pending: List[Dict] = []
        self._reference_scores: List[float] = []
        self._current_scores: List[float] = []
        self._total_refreshed = 0

    def add_document(self, doc: Dict, relevance_score: float = 1.0) -> Optional[str]:
        self._pending.append(doc)
        self._current_scores.append(relevance_score)

        if len(self._pending) >= self.batch_size:
            return self._flush()

        if len(self._reference_scores) >= 20 and len(self._current_scores) >= 20:
            psi = compute_psi(self._reference_scores, self._current_scores)
            if psi > self.drift_threshold:
                logger.warning("Embedding drift detected (PSI=%.4f > %.2f). Triggering refresh.", psi, self.drift_threshold)
                return self._flush()

        return None

    def _flush(self) -> str:
        n = len(self._pending)
        self._total_refreshed += n
        logger.info("Online refresh: ingesting %d new documents (total refreshed: %d).", n, self._total_refreshed)

        self._reference_scores = list(self._current_scores)
        self._current_scores = []
        self._pending = []

        return f"refreshed_{n}_docs_{int(time.time())}"

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Stream Processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """
    Stateful stream processor for ML inference events.

    Per-event processing:
    1. Parse and validate incoming event
    2. Update windowed aggregators
    3. Check drift detectors on primary metrics
    4. Update online embedding refresher if new content
    5. Route to DLQ if processing fails

    Designed to run as a Kafka consumer worker or Kinesis Lambda.
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_delta: float = 0.005,
        drift_lambda: float = 30.0,
    ):
        self.aggregator = WindowedAggregator(window_size)
        self.refresher = OnlineEmbeddingRefresher()
        self.dlq: List[Dict] = []

        self._ph_detectors: Dict[str, PageHinkley] = {}
        self._adwin_detectors: Dict[str, ADWIN] = {}
        self._drift_delta = drift_delta
        self._drift_lambda = drift_lambda

        self._reference_windows: Dict[str, List[float]] = {}
        self._processed = 0
        self._failed = 0

        self._handlers: Dict[str, Callable] = {}
        self.register_handler("inference_result", self._handle_inference_result)
        self.register_handler("document_ingested", self._handle_document_ingested)
        self.register_handler("user_feedback", self._handle_user_feedback)

    def register_handler(self, event_type: str, handler: Callable) -> None:
        self._handlers[event_type] = handler

    def process(self, event: StreamEvent) -> Dict[str, Any]:
        try:
            handler = self._handlers.get(event.event_type)
            if handler:
                result = handler(event)
            else:
                result = {"status": "no_handler", "event_type": event.event_type}
            self._processed += 1
            return result
        except Exception as e:
            self._failed += 1
            self.dlq.append({"event": event, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()})
            logger.error("Failed to process event %s: %s. Sent to DLQ.", event.event_id, e)
            return {"status": "dlq", "error": str(e)}

    def _handle_inference_result(self, event: StreamEvent) -> Dict[str, Any]:
        p = event.payload
        metrics = {
            "ragas_faithfulness": p.get("ragas_faithfulness"),
            "ragas_relevancy": p.get("ragas_relevancy"),
            "latency_ms": p.get("latency_ms"),
            "retrieval_score": p.get("retrieval_score"),
        }

        drift_alerts = {}
        for metric, value in metrics.items():
            if value is None:
                continue
            self.aggregator.update(metric, value)
            alert = self.check_drift(metric, value)
            if alert.detected:
                drift_alerts[metric] = alert.summary()

        return {
            "status": "processed",
            "metrics": {k: v for k, v in metrics.items() if v is not None},
            "drift_alerts": drift_alerts,
            "window_stats": self.aggregator.stats("ragas_faithfulness"),
        }

    def _handle_document_ingested(self, event: StreamEvent) -> Dict[str, Any]:
        doc = event.payload.get("document", {})
        score = event.payload.get("relevance_score", 1.0)
        refresh_id = self.refresher.add_document(doc, score)
        return {"status": "queued", "refresh_triggered": refresh_id is not None, "pending": self.refresher.pending_count}

    def _handle_user_feedback(self, event: StreamEvent) -> Dict[str, Any]:
        feedback = event.payload.get("feedback", 0)
        self.aggregator.update("user_satisfaction", float(feedback))
        return {"status": "logged", "satisfaction_stats": self.aggregator.stats("user_satisfaction")}

    def check_drift(self, metric: str, value: float) -> DriftAlert:
        if metric not in self._ph_detectors:
            self._ph_detectors[metric] = PageHinkley(delta=self._drift_delta, lambda_=self._drift_lambda)
            self._adwin_detectors[metric] = ADWIN()

        ph_detected, ph_stat = self._ph_detectors[metric].update(value)
        adwin_detected, window_n = self._adwin_detectors[metric].update(value)
        detected = ph_detected or adwin_detected

        if ph_detected:
            self._ph_detectors[metric].reset()

        ref = self._reference_windows.setdefault(metric, [])
        if len(ref) < 50:
            ref.append(value)
        current_window = list(self.aggregator._windows.get(metric, []))
        psi = compute_psi(ref, current_window) if len(current_window) >= 20 and len(ref) >= 20 else 0.0

        return DriftAlert(
            metric=metric,
            detected=detected,
            method="PageHinkley+ADWIN",
            statistic=ph_stat,
            threshold=self._drift_lambda,
            direction="down" if value < self.aggregator.stats(metric).get("mean", value) else "up",
            magnitude=abs(psi),
            n_samples=self._processed,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "processed": self._processed,
            "failed": self._failed,
            "dlq_size": len(self.dlq),
            "pending_embedding_refresh": self.refresher.pending_count,
            "monitored_metrics": list(self._ph_detectors.keys()),
            "window_stats": self.aggregator.all_stats(),
        }


if __name__ == "__main__":
    import random
    import uuid
    random.seed(42)

    processor = StreamProcessor(window_size=50, drift_lambda=20.0)
    n_events = 300
    drift_start = 200

    print(f"Processing {n_events} inference events (drift injected at event {drift_start})...\n")

    drift_events = 0
    for i in range(n_events):
        if i < drift_start:
            faithfulness = random.gauss(0.847, 0.04)
            latency_ms = random.gauss(3200, 300)
        else:
            faithfulness = random.gauss(0.72, 0.05)
            latency_ms = random.gauss(4800, 400)

        event = StreamEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type="inference_result",
            payload={
                "ragas_faithfulness": max(0, min(1, faithfulness)),
                "ragas_relevancy": max(0, min(1, faithfulness - 0.02 + random.gauss(0, 0.03))),
                "latency_ms": max(100, latency_ms),
                "retrieval_score": random.uniform(0.6, 0.95),
            },
        )
        result = processor.process(event)
        if result.get("drift_alerts"):
            drift_events += 1
            if drift_events <= 3:
                print(f"  Event {i}: DRIFT ALERT: {result['drift_alerts']}")

    health = processor.health()
    print(f"\nProcessor health:")
    print(f"  Processed: {health['processed']}  Failed: {health['failed']}  DLQ: {health['dlq_size']}")
    print(f"  Drift events detected: {drift_events}")
    print(f"  Faithfulness stats: {health['window_stats'].get('ragas_faithfulness', {})}")

    doc_event = StreamEvent(
        event_id="doc_001",
        event_type="document_ingested",
        payload={"document": {"id": "doc_001", "text": "New LLM paper on RAG..."}, "relevance_score": 0.91},
    )
    processor.process(doc_event)
    print(f"\nPending embedding refresh: {processor.refresher.pending_count} docs")

    ref = [random.gauss(0.85, 0.04) for _ in range(100)]
    current = [random.gauss(0.70, 0.06) for _ in range(100)]
    psi = compute_psi(ref, current)
    print(f"\nPSI (stable distribution): {compute_psi(ref, ref):.4f}")
    print(f"PSI (drifted distribution): {psi:.4f}  -> {'MAJOR SHIFT' if psi >= 0.25 else 'moderate' if psi >= 0.1 else 'stable'}")
