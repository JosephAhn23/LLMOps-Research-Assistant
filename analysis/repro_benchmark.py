"""
Reproducible benchmark runner for latency/throughput evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _offline_call(query: str) -> None:
    import agents.orchestrator as orchestrator

    class _FakeRetriever:
        def retrieve(self, q: str) -> list[dict]:
            return [
                {
                    "text": f"chunk-{i} for {q}",
                    "source": f"doc-{i}.md",
                    "retrieval_score": 1.0 / (i + 1),
                }
                for i in range(12)
            ]

    class _FakeReranker:
        def rerank(self, _q: str, candidates: list[dict]) -> list[dict]:
            reranked = []
            for c in candidates:
                enriched = dict(c)
                enriched["rerank_score"] = enriched.get("retrieval_score", 0.0)
                reranked.append(enriched)
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:5]

    class _FakeSynthesizer:
        def synthesize(self, q: str, chunks: list[dict]) -> dict:
            return {
                "answer": f"offline answer for: {q}",
                "sources": [c["source"] for c in chunks],
                "tokens_used": 128,
                "prompt_tokens": 96,
                "completion_tokens": 32,
            }

    orchestrator.retriever = _FakeRetriever()
    orchestrator.reranker = _FakeReranker()
    orchestrator.synthesizer = _FakeSynthesizer()

    result = orchestrator.run_pipeline(query)
    if result.get("error"):
        raise RuntimeError(result["error"])


def _api_call(url: str, query: str, timeout_s: float) -> None:
    payload = {"query": query, "stream": False}
    resp = httpx.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data["error"])


def run_benchmark(
    mode: str,
    runs: int,
    concurrency: int,
    query: str,
    api_url: str,
    timeout_s: float,
) -> dict:
    latencies_ms: list[float] = []
    failures: list[str] = []
    started_at = time.perf_counter()

    if mode == "offline_pipeline":
        call: Callable[[], None] = lambda: _offline_call(query)
    else:
        call = lambda: _api_call(api_url, query, timeout_s)

    def timed_call() -> float:
        t0 = time.perf_counter()
        call()
        return (time.perf_counter() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(timed_call) for _ in range(runs)]
        for fut in as_completed(futures):
            try:
                latencies_ms.append(fut.result())
            except Exception as exc:
                failures.append(str(exc))

    elapsed_s = time.perf_counter() - started_at
    successes = len(latencies_ms)
    total = runs
    throughput_qps = successes / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "mode": mode,
        "runs_requested": runs,
        "runs_succeeded": successes,
        "runs_failed": len(failures),
        "success_rate": successes / total if total else 0.0,
        "concurrency": concurrency,
        "latency_ms": {
            "min": min(latencies_ms) if latencies_ms else 0.0,
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "throughput_qps": throughput_qps,
        "elapsed_s": elapsed_s,
        "failure_samples": failures[:5],
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_sha": _git_sha(),
        },
    }


def _write_results(output_root: Path, metrics: dict) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = [
        "# Reproducible Benchmark Report",
        "",
        f"- Mode: `{metrics['mode']}`",
        f"- Git SHA: `{metrics['metadata']['git_sha']}`",
        f"- Python: `{metrics['metadata']['python']}`",
        f"- Platform: `{metrics['metadata']['platform']}`",
        f"- Runs: `{metrics['runs_succeeded']}/{metrics['runs_requested']}`",
        f"- Concurrency: `{metrics['concurrency']}`",
        f"- Throughput (QPS): `{metrics['throughput_qps']:.3f}`",
        f"- p50 latency (ms): `{metrics['latency_ms']['p50']:.2f}`",
        f"- p95 latency (ms): `{metrics['latency_ms']['p95']:.2f}`",
        f"- Max latency (ms): `{metrics['latency_ms']['max']:.2f}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible benchmark suite.")
    parser.add_argument("--mode", choices=["offline_pipeline", "api"], default="offline_pipeline")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--query", default="How does distributed retrieval work?")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/query")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--output-dir", default="analysis/benchmarks")
    args = parser.parse_args()

    metrics = run_benchmark(
        mode=args.mode,
        runs=max(1, args.runs),
        concurrency=max(1, args.concurrency),
        query=args.query,
        api_url=args.api_url,
        timeout_s=max(1.0, args.timeout_s),
    )
    out_dir = _write_results(Path(args.output_dir), metrics)
    print(f"Wrote benchmark artifacts to: {out_dir}")
    print(json.dumps(metrics["latency_ms"], indent=2))


if __name__ == "__main__":
    main()
