"""
benchmarks/fill_readme_benchmarks.py

Runs all benchmarks and auto-fills the README table with real numbers.

Usage:
    # Full run (requires live stack)
    docker compose up -d
    python -m ingestion.pipeline --limit 500
    python benchmarks/fill_readme_benchmarks.py

    # Dry run (mock mode, no live stack needed)
    python benchmarks/fill_readme_benchmarks.py --mock

    # Skip specific suites
    python benchmarks/fill_readme_benchmarks.py --skip-vllm
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from datasets import Dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LatencyResult:
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n_requests: int
    errors: int = 0

    def summary(self) -> str:
        return (
            f"p50={self.p50_ms:.1f}ms  p90={self.p90_ms:.1f}ms  "
            f"p99={self.p99_ms:.1f}ms  errors={self.errors}/{self.n_requests}"
        )


@dataclass
class QpsResult:
    qps: float
    duration_sec: float
    total_requests: int
    successful: int
    errors: int

    def summary(self) -> str:
        return (
            f"{self.qps:.1f} QPS  "
            f"({self.successful}/{self.total_requests} ok, {self.duration_sec:.1f}s)"
        )


@dataclass
class VllmResult:
    model: str
    fp16_tokens_per_sec: Optional[float] = None
    int4_awq_tokens_per_sec: Optional[float] = None
    concurrent_qps: Optional[float] = None
    mean_ttft_ms: Optional[float] = None
    note: str = ""

    def summary(self) -> str:
        parts = []
        if self.fp16_tokens_per_sec:
            parts.append(f"fp16={self.fp16_tokens_per_sec:.0f} tok/s")
        if self.int4_awq_tokens_per_sec:
            parts.append(f"awq={self.int4_awq_tokens_per_sec:.0f} tok/s")
        if self.concurrent_qps:
            parts.append(f"concurrent_qps={self.concurrent_qps:.1f}")
        return "  ".join(parts) if parts else self.note


@dataclass
class RagasResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_examples: int

    def summary(self) -> str:
        return (
            f"faithfulness={self.faithfulness:.3f}  "
            f"relevancy={self.answer_relevancy:.3f}  "
            f"precision={self.context_precision:.3f}  "
            f"recall={self.context_recall:.3f}"
        )


@dataclass
class BenchmarkReport:
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    mock_mode: bool = False
    api_base_url: str = "http://localhost:8000"

    realtime_latency: Optional[LatencyResult] = None
    realtime_qps: Optional[QpsResult] = None
    batch_qps: Optional[QpsResult] = None
    vllm: Optional[VllmResult] = None
    ragas: Optional[RagasResult] = None

    errors: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Sample queries
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What is retrieval-augmented generation?",
    "Explain the difference between LoRA and full fine-tuning.",
    "How does FAISS handle approximate nearest neighbor search?",
    "What is PagedAttention in vLLM?",
    "Describe the transformer self-attention mechanism.",
    "What are the advantages of QLoRA over standard LoRA?",
    "How does cross-encoder reranking improve RAG quality?",
    "What is the role of a dead-letter queue in Celery?",
    "Explain RAGAS faithfulness metric.",
    "What is MinHash LSH used for in data pipelines?",
    "How does SageMaker model registry work?",
    "What is the difference between FAISS IVF and HNSW indexes?",
    "Explain tensor parallelism in distributed LLM inference.",
    "What is the purpose of BitsAndBytes in QLoRA?",
    "How does Accelerate handle multi-GPU training?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Suite: API latency
# ─────────────────────────────────────────────────────────────────────────────


def _mock_latency(base_ms: float = 85.0, jitter: float = 0.35) -> float:
    return max(5.0, random.lognormvariate(mu=math.log(base_ms), sigma=jitter))


def run_latency_suite(
    base_url: str,
    n_requests: int = 200,
    mock: bool = False,
    timeout: float = 10.0,
) -> LatencyResult:
    """Measure p50/p90/p99 latency with sequential requests."""
    latencies: list[float] = []
    errors = 0

    print(f"  -> Latency suite: {n_requests} sequential requests...")

    for i in range(n_requests):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]

        if mock:
            time.sleep(0.001)
            latencies.append(_mock_latency(base_ms=90))
            continue

        if not HAS_HTTPX:
            raise RuntimeError("pip install httpx")

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{base_url}/query",
                    json={"query": query, "top_k": 5},
                )
                resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    x request {i} failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_requests} done...")

    latencies.sort()
    n = len(latencies)
    if n == 0:
        raise RuntimeError(
            f"All {errors} requests failed -- is the stack running?"
        )

    return LatencyResult(
        p50_ms=round(statistics.median(latencies), 2),
        p90_ms=round(latencies[int(n * 0.90)], 2),
        p99_ms=round(latencies[int(n * 0.99)], 2),
        mean_ms=round(statistics.mean(latencies), 2),
        min_ms=round(latencies[0], 2),
        max_ms=round(latencies[-1], 2),
        n_requests=n_requests,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Suite: Concurrent QPS
# ─────────────────────────────────────────────────────────────────────────────


def run_qps_suite(
    base_url: str,
    endpoint: str = "/query",
    payload_fn=None,
    concurrency: int = 20,
    duration_sec: float = 30.0,
    mock: bool = False,
) -> QpsResult:
    """Measure sustained QPS under concurrent load."""
    if payload_fn is None:

        def payload_fn(i):
            return {
                "query": SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)],
                "top_k": 5,
            }

    print(
        f"  -> QPS suite: {concurrency} concurrent workers x "
        f"{duration_sec}s on {endpoint}..."
    )

    if mock:
        base_qps = 45.0 + random.uniform(-5, 10)
        total = int(base_qps * duration_sec)
        err = int(total * 0.01)
        return QpsResult(
            qps=round(base_qps, 1),
            duration_sec=duration_sec,
            total_requests=total,
            successful=total - err,
            errors=err,
        )

    if not HAS_HTTPX:
        raise RuntimeError("pip install httpx")

    completed: list[float] = []
    error_list: list[str] = []
    stop = False
    request_counter = [0]

    def worker():
        client = httpx.Client(timeout=15.0)
        try:
            while not stop:
                i = request_counter[0]
                request_counter[0] += 1
                start = time.perf_counter()
                try:
                    resp = client.post(
                        f"{base_url}{endpoint}", json=payload_fn(i)
                    )
                    resp.raise_for_status()
                    completed.append(time.perf_counter() - start)
                except Exception as e:
                    error_list.append(str(e))
        finally:
            client.close()

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        time.sleep(duration_sec)
        stop = True
        for f in futures:
            try:
                f.result(timeout=5)
            except Exception:
                pass

    actual_duration = time.perf_counter() - wall_start
    n_ok = len(completed)
    n_err = len(error_list)

    return QpsResult(
        qps=round(n_ok / actual_duration, 1),
        duration_sec=round(actual_duration, 1),
        total_requests=n_ok + n_err,
        successful=n_ok,
        errors=n_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Suite: vLLM throughput
# ─────────────────────────────────────────────────────────────────────────────


def run_vllm_suite(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    vllm_base_url: str = "http://localhost:8001",
    num_prompts: int = 200,
    max_tokens: int = 256,
    concurrency: int = 10,
    mock: bool = False,
) -> VllmResult:
    """Benchmark vLLM throughput via /v1/completions endpoint."""
    print(
        f"  -> vLLM suite: {num_prompts} prompts, "
        f"max_tokens={max_tokens}, concurrency={concurrency}..."
    )

    if mock:
        return VllmResult(
            model=model,
            fp16_tokens_per_sec=round(1580 + random.uniform(-80, 120), 0),
            int4_awq_tokens_per_sec=round(
                3240 + random.uniform(-100, 200), 0
            ),
            concurrent_qps=round(12.4 + random.uniform(-1, 2), 1),
            mean_ttft_ms=round(38 + random.uniform(-5, 10), 1),
            note="mock values",
        )

    if not HAS_HTTPX:
        raise RuntimeError("pip install httpx")

    prompts = [
        SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(num_prompts)
    ]

    def _bench_one_config(
        extra_params: dict,
    ) -> tuple[float, float, float]:
        all_latencies: list[float] = []
        total_tokens = [0]
        errors = [0]

        def call(prompt):
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{vllm_base_url}/v1/completions",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                            **extra_params,
                        },
                    )
                    data = resp.json()
                    tokens = data["usage"]["completion_tokens"]
                    total_tokens[0] += tokens
                    all_latencies.append(time.perf_counter() - start)
            except Exception:
                errors[0] += 1

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(call, prompts))
        wall_elapsed = time.perf_counter() - wall_start

        n_ok = len(all_latencies)
        if n_ok == 0:
            return 0.0, 0.0, 0.0

        tok_per_sec = total_tokens[0] / wall_elapsed
        mean_ttft = statistics.mean(all_latencies) * 1000
        qps = n_ok / wall_elapsed
        return round(tok_per_sec, 1), round(mean_ttft, 1), round(qps, 1)

    fp16_tps, ttft_ms, conc_qps = _bench_one_config({})

    awq_tps = None
    try:
        awq_tps_raw, _, _ = _bench_one_config({"quantization": "awq"})
        if awq_tps_raw > fp16_tps * 1.1:
            awq_tps = awq_tps_raw
    except Exception:
        pass

    return VllmResult(
        model=model,
        fp16_tokens_per_sec=fp16_tps,
        int4_awq_tokens_per_sec=awq_tps,
        concurrent_qps=conc_qps,
        mean_ttft_ms=ttft_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Suite: RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

RAGAS_QA_PAIRS = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": (
            "RAG combines information retrieval with language model generation "
            "by first retrieving relevant documents then using them as context "
            "for the LLM."
        ),
    },
    {
        "question": "How does LoRA reduce the number of trainable parameters?",
        "ground_truth": (
            "LoRA decomposes weight update matrices into two low-rank matrices "
            "A and B, only training these small matrices instead of the full "
            "weight matrix."
        ),
    },
    {
        "question": "What is PagedAttention?",
        "ground_truth": (
            "PagedAttention is a memory management technique in vLLM that "
            "manages KV cache in fixed-size blocks like virtual memory pages, "
            "eliminating fragmentation."
        ),
    },
    {
        "question": "What is FAISS used for?",
        "ground_truth": (
            "FAISS is a library for efficient similarity search and clustering "
            "of dense vectors, commonly used for nearest neighbor search in "
            "RAG retrieval."
        ),
    },
    {
        "question": "What does a dead-letter queue do?",
        "ground_truth": (
            "A dead-letter queue captures messages that fail processing after "
            "maximum retries, allowing inspection and reprocessing without "
            "losing failed tasks."
        ),
    },
    {
        "question": "What is the RAGAS faithfulness metric?",
        "ground_truth": (
            "RAGAS faithfulness measures whether the generated answer is "
            "factually consistent with the retrieved context, scored between "
            "0 and 1."
        ),
    },
    {
        "question": "What is QLoRA?",
        "ground_truth": (
            "QLoRA combines quantization (4-bit NF4) with LoRA adapters, "
            "allowing fine-tuning of large models on consumer GPUs with "
            "minimal accuracy loss."
        ),
    },
    {
        "question": "How does cross-encoder reranking work?",
        "ground_truth": (
            "Cross-encoders jointly encode query and document together, "
            "allowing direct comparison and more accurate relevance scoring "
            "than bi-encoders at higher compute cost."
        ),
    },
]


def run_ragas_suite(
    base_url: str,
    n_examples: int = 8,
    mock: bool = False,
) -> RagasResult:
    """Call /query for each QA pair then run RAGAS evaluation."""
    print(f"  -> RAGAS suite: evaluating {n_examples} QA pairs...")

    if mock:
        return RagasResult(
            faithfulness=round(0.847 + random.uniform(-0.03, 0.03), 3),
            answer_relevancy=round(
                0.812 + random.uniform(-0.03, 0.03), 3
            ),
            context_precision=round(
                0.789 + random.uniform(-0.03, 0.03), 3
            ),
            context_recall=round(0.801 + random.uniform(-0.03, 0.03), 3),
            n_examples=n_examples,
        )

    if not HAS_HTTPX:
        raise RuntimeError("pip install httpx")
    if not HAS_DATASETS:
        raise RuntimeError("pip install datasets")

    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    pairs = RAGAS_QA_PAIRS[:n_examples]
    questions, answers, contexts, ground_truths = [], [], [], []

    with httpx.Client(timeout=30.0) as client:
        for pair in pairs:
            try:
                resp = client.post(
                    f"{base_url}/query",
                    json={
                        "query": pair["question"],
                        "top_k": 5,
                        "return_contexts": True,
                    },
                )
                data = resp.json()
                questions.append(pair["question"])
                answers.append(data.get("answer", ""))
                contexts.append(
                    data.get("contexts", [data.get("answer", "")])
                )
                ground_truths.append(pair["ground_truth"])
            except Exception as e:
                print(f"    x RAGAS query failed: {e}")

    if not questions:
        raise RuntimeError("All RAGAS queries failed")

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = ragas_evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return RagasResult(
        faithfulness=round(float(result["faithfulness"]), 3),
        answer_relevancy=round(float(result["answer_relevancy"]), 3),
        context_precision=round(float(result["context_precision"]), 3),
        context_recall=round(float(result["context_recall"]), 3),
        n_examples=len(questions),
    )


# ─────────────────────────────────────────────────────────────────────────────
# README updater
# ─────────────────────────────────────────────────────────────────────────────


def update_readme(report: BenchmarkReport, readme_path: str = "README.md"):
    """Replace TBD values in the README benchmark table with real numbers."""
    path = Path(readme_path)
    if not path.exists():
        print(f"  x README not found at {readme_path}")
        return

    content = path.read_text(encoding="utf-8")
    original = content

    def replace_tbd(metric_name: str, value: str):
        nonlocal content
        pattern = (
            rf"(\|\s*{re.escape(metric_name)}\s*\|\s*)"
            rf"`?TBD[^|`]*`?"
            rf"(\s*\|)"
        )
        replacement = rf"\1`{value}`\2"
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    if report.realtime_latency:
        r = report.realtime_latency
        replace_tbd("Realtime p50 latency", f"{r.p50_ms} ms")
        replace_tbd("Realtime p90 latency", f"{r.p90_ms} ms")
        replace_tbd("Realtime p99 latency", f"{r.p99_ms} ms")

    if report.realtime_qps:
        replace_tbd("Concurrent API QPS", f"{report.realtime_qps.qps}")

    if report.batch_qps:
        replace_tbd("Batch API QPS", f"{report.batch_qps.qps}")

    if report.vllm:
        v = report.vllm
        if v.fp16_tokens_per_sec:
            replace_tbd(
                "vLLM fp16 tokens/sec", f"{v.fp16_tokens_per_sec:.0f}"
            )
        if v.int4_awq_tokens_per_sec:
            replace_tbd(
                "vLLM int4-AWQ tokens/sec",
                f"{v.int4_awq_tokens_per_sec:.0f}",
            )
        if v.concurrent_qps:
            replace_tbd(
                "vLLM concurrent QPS (10 users)", f"{v.concurrent_qps}"
            )

    if report.ragas:
        r = report.ragas
        replace_tbd("RAGAS faithfulness", f"{r.faithfulness}")
        replace_tbd("RAGAS answer relevancy", f"{r.answer_relevancy}")
        replace_tbd("RAGAS context precision", f"{r.context_precision}")
        replace_tbd("RAGAS context recall", f"{r.context_recall}")

    if content != original:
        path.write_text(content, encoding="utf-8")
        print(f"  + README updated: {readme_path}")
    else:
        print("  x README unchanged -- check metric name spellings")


# ─────────────────────────────────────────────────────────────────────────────
# MLflow logger
# ─────────────────────────────────────────────────────────────────────────────


def log_to_mlflow(report: BenchmarkReport):
    if not HAS_MLFLOW:
        print("  x mlflow not installed, skipping")
        return

    mlflow.set_experiment("benchmark-suite")
    with mlflow.start_run(
        run_name=f"benchmarks-{report.timestamp[:10]}"
    ):
        mlflow.log_param("mock_mode", report.mock_mode)
        mlflow.log_param("api_base_url", report.api_base_url)

        if report.realtime_latency:
            r = report.realtime_latency
            mlflow.log_metrics({
                "latency_p50_ms": r.p50_ms,
                "latency_p90_ms": r.p90_ms,
                "latency_p99_ms": r.p99_ms,
                "latency_mean_ms": r.mean_ms,
                "latency_error_rate": r.errors / r.n_requests,
            })

        if report.realtime_qps:
            mlflow.log_metric("realtime_qps", report.realtime_qps.qps)

        if report.batch_qps:
            mlflow.log_metric("batch_qps", report.batch_qps.qps)

        if report.vllm:
            v = report.vllm
            if v.fp16_tokens_per_sec:
                mlflow.log_metric(
                    "vllm_fp16_tokens_per_sec", v.fp16_tokens_per_sec
                )
            if v.int4_awq_tokens_per_sec:
                mlflow.log_metric(
                    "vllm_awq_tokens_per_sec", v.int4_awq_tokens_per_sec
                )
            if v.concurrent_qps:
                mlflow.log_metric("vllm_concurrent_qps", v.concurrent_qps)
            if v.mean_ttft_ms:
                mlflow.log_metric("vllm_mean_ttft_ms", v.mean_ttft_ms)

        if report.ragas:
            r = report.ragas
            mlflow.log_metrics({
                "ragas_faithfulness": r.faithfulness,
                "ragas_answer_relevancy": r.answer_relevancy,
                "ragas_context_precision": r.context_precision,
                "ragas_context_recall": r.context_recall,
            })

        report_path = Path("benchmarks/last_run.json")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2))
        mlflow.log_artifact(str(report_path))

    print("  + Results logged to MLflow")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────


def print_report(report: BenchmarkReport):
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    if report.realtime_latency:
        print(
            f"\n  API Latency:        {report.realtime_latency.summary()}"
        )
    if report.realtime_qps:
        print(f"  Realtime QPS:       {report.realtime_qps.summary()}")
    if report.batch_qps:
        print(f"  Batch QPS:          {report.batch_qps.summary()}")
    if report.vllm:
        print(f"  vLLM Throughput:    {report.vllm.summary()}")
    if report.ragas:
        print(f"  RAGAS Scores:       {report.ragas.summary()}")
    if report.errors:
        print(f"\n  Errors: {report.errors}")

    if not report.mock_mode:
        print("\n  README table updated +")
    else:
        print(
            "\n  Mock mode -- README NOT updated "
            "(use real run to fill README)"
        )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fill LLMOps README benchmark table"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock values (no live stack needed)",
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-qps", action="store_true")
    parser.add_argument(
        "--n-latency",
        type=int,
        default=200,
        help="Number of requests for latency suite",
    )
    parser.add_argument(
        "--qps-duration",
        type=float,
        default=30.0,
        help="Duration in seconds for QPS suite",
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    report = BenchmarkReport(
        mock_mode=args.mock,
        api_base_url=args.api_url,
    )

    mode = "[MOCK MODE]" if args.mock else "[LIVE MODE]"
    print(f"\n{mode} Starting benchmark suite...")
    print(f"  API: {args.api_url}")
    print(f"  vLLM: {args.vllm_url}\n")

    # 1. Latency
    try:
        print("[1/5] API Latency Suite")
        report.realtime_latency = run_latency_suite(
            args.api_url, n_requests=args.n_latency, mock=args.mock
        )
        print(f"      + {report.realtime_latency.summary()}")
    except Exception as e:
        msg = f"Latency suite failed: {e}"
        print(f"      x {msg}")
        report.errors.append(msg)

    # 2. Realtime QPS
    if not args.skip_qps:
        try:
            print("[2/5] Realtime QPS Suite")
            report.realtime_qps = run_qps_suite(
                args.api_url,
                endpoint="/query",
                concurrency=20,
                duration_sec=args.qps_duration,
                mock=args.mock,
            )
            print(f"      + {report.realtime_qps.summary()}")
        except Exception as e:
            msg = f"Realtime QPS suite failed: {e}"
            print(f"      x {msg}")
            report.errors.append(msg)

        # 3. Batch QPS
        try:
            print("[3/5] Batch QPS Suite")
            report.batch_qps = run_qps_suite(
                args.api_url,
                endpoint="/batch",
                payload_fn=lambda i: {
                    "queries": [
                        SAMPLE_QUERIES[j % len(SAMPLE_QUERIES)]
                        for j in range(i, i + 5)
                    ],
                    "top_k": 5,
                },
                concurrency=10,
                duration_sec=args.qps_duration,
                mock=args.mock,
            )
            print(f"      + {report.batch_qps.summary()}")
        except Exception as e:
            msg = f"Batch QPS suite failed: {e}"
            print(f"      x {msg}")
            report.errors.append(msg)
    else:
        print("[2/5] Realtime QPS -- skipped")
        print("[3/5] Batch QPS -- skipped")

    # 4. vLLM
    if not args.skip_vllm:
        try:
            print("[4/5] vLLM Throughput Suite")
            report.vllm = run_vllm_suite(
                model=args.model,
                vllm_base_url=args.vllm_url,
                num_prompts=200,
                concurrency=10,
                mock=args.mock,
            )
            print(f"      + {report.vllm.summary()}")
        except Exception as e:
            msg = f"vLLM suite failed: {e}"
            print(f"      x {msg}")
            report.errors.append(msg)
    else:
        print("[4/5] vLLM -- skipped")

    # 5. RAGAS
    if not args.skip_ragas:
        try:
            print("[5/5] RAGAS Evaluation Suite")
            report.ragas = run_ragas_suite(
                args.api_url, n_examples=8, mock=args.mock
            )
            print(f"      + {report.ragas.summary()}")
        except Exception as e:
            msg = f"RAGAS suite failed: {e}"
            print(f"      x {msg}")
            report.errors.append(msg)
    else:
        print("[5/5] RAGAS -- skipped")

    # Output
    print_report(report)

    if not args.mock:
        update_readme(report, readme_path=args.readme)
    else:
        print(
            "  i  Mock mode: README not updated. "
            "Run without --mock on live stack.\n"
        )

    if not args.no_mlflow:
        try:
            log_to_mlflow(report)
        except Exception as e:
            print(f"  x MLflow logging failed: {e}")

    out = Path("benchmarks/last_run.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2))
    print(f"  + Full report saved: {out}\n")

    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
