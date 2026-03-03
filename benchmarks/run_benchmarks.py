"""
Latency + throughput benchmarks - generates real numbers for resume.
Run this once to populate your README metrics.
"""
import asyncio
import json
import statistics
import time
from typing import Dict

import httpx

API_URL = "http://localhost:8000"

TEST_QUERIES = [
    "What are the main applications of transformer architectures?",
    "How does retrieval augmented generation improve LLM accuracy?",
    "What is the difference between fine-tuning and prompt engineering?",
    "Explain the attention mechanism in neural networks.",
    "What are the trade-offs between FAISS IVF and HNSW indexes?",
    "How does LoRA reduce the number of trainable parameters?",
    "What is the purpose of cross-encoder reranking in RAG pipelines?",
    "Describe the role of MLflow in experiment tracking.",
    "How does distributed training differ from data parallelism?",
    "What metrics are used to evaluate RAG pipeline quality?",
]


def benchmark_realtime_latency(n_runs: int = 50) -> Dict:
    """Measure p50/p90/p99 latency for single queries."""
    latencies = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(n_runs):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            start = time.perf_counter()
            response = client.post(f"{API_URL}/query", json={"query": query})
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                latencies.append(latency_ms)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_runs} queries completed")

    latencies.sort()
    return {
        "n_requests": len(latencies),
        "p50_ms": round(statistics.median(latencies), 1),
        "p90_ms": round(latencies[int(len(latencies) * 0.90)], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


async def benchmark_concurrent_throughput(
    n_concurrent: int = 20,
    n_total: int = 200,
) -> Dict:
    """Measure throughput under concurrent load."""
    semaphore = asyncio.Semaphore(n_concurrent)
    latencies = []
    errors = 0

    async def single_request(client: httpx.AsyncClient, query: str):
        nonlocal errors
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await client.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=30.0,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    latencies.append(latency_ms)
                else:
                    errors += 1
            except Exception:
                errors += 1

    start_total = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [
            single_request(client, TEST_QUERIES[i % len(TEST_QUERIES)])
            for i in range(n_total)
        ]
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_total
    qps = len(latencies) / total_time

    return {
        "n_concurrent": n_concurrent,
        "n_total": n_total,
        "successful": len(latencies),
        "errors": errors,
        "error_rate": round(errors / n_total, 3),
        "total_time_s": round(total_time, 2),
        "queries_per_second": round(qps, 1),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
    }


def benchmark_batch_throughput(n_queries: int = 100) -> Dict:
    """Measure batch inference throughput."""
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)] for i in range(n_queries)]

    with httpx.Client(timeout=120.0) as client:
        start = time.perf_counter()
        response = client.post(f"{API_URL}/batch", json={"queries": queries})
        job_id = response.json()["job_id"]

        while True:
            status = client.get(f"{API_URL}/batch/{job_id}").json()
            if status.get("status") in ["completed", "failed"]:
                break
            if float(status.get("completion_pct", 0)) >= 100:
                break
            time.sleep(0.5)

        total_time = time.perf_counter() - start

    return {
        "n_queries": n_queries,
        "total_time_s": round(total_time, 2),
        "queries_per_second": round(n_queries / total_time, 1),
    }


def run_all_benchmarks():
    print("=== LLMOps Research Assistant Benchmarks ===\n")

    print("1. Realtime latency (50 sequential requests)...")
    latency_results = benchmark_realtime_latency(n_runs=50)
    print(f"   p50: {latency_results['p50_ms']}ms")
    print(f"   p90: {latency_results['p90_ms']}ms")
    print(f"   p99: {latency_results['p99_ms']}ms\n")

    print("2. Concurrent throughput (20 concurrent, 200 total)...")
    throughput = asyncio.run(benchmark_concurrent_throughput(20, 200))
    print(f"   QPS: {throughput['queries_per_second']}")
    print(f"   Error rate: {throughput['error_rate']}")
    print(f"   p99 latency: {throughput['p99_ms']}ms\n")

    print("3. Batch throughput (100 queries)...")
    batch = benchmark_batch_throughput(100)
    print(f"   QPS: {batch['queries_per_second']}\n")

    results = {
        "latency": latency_results,
        "throughput": throughput,
        "batch": batch,
    }

    with open("benchmarks/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Summary (paste into README) ===")
    print(f"| Realtime p50 latency  | {latency_results['p50_ms']}ms |")
    print(f"| Realtime p99 latency  | {latency_results['p99_ms']}ms |")
    print(f"| Concurrent QPS        | {throughput['queries_per_second']} |")
    print(f"| Batch QPS             | {batch['queries_per_second']} |")
    print(f"| Error rate            | {throughput['error_rate']} |")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
