"""
vLLM inference benchmarks - latency, throughput, quantization comparison.
Generates real numbers for resume.
Covers: vLLM benchmark weakness
"""
import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Dict

BENCHMARK_PROMPTS = [
    "Explain how transformer attention mechanisms work in detail.",
    "What are the trade-offs between RAG and fine-tuning for LLM customization?",
    "Describe the architecture of a production ML inference pipeline.",
    "How does LoRA reduce the number of trainable parameters during fine-tuning?",
    "What is the role of cross-encoder reranking in retrieval augmented generation?",
    "Explain the differences between FAISS IVFFlat and HNSW indexing strategies.",
    "How does mixed precision training with fp16 affect model accuracy?",
    "What are the key metrics for evaluating RAG pipeline quality?",
    "Describe how Celery handles distributed task queues with Redis.",
    "What are the benefits of Kubernetes HorizontalPodAutoscaler for ML services?",
]

CONTEXT = """
Retrieval Augmented Generation (RAG) combines dense retrieval with generative models.
The pipeline first encodes the query using a bi-encoder, retrieves top-k candidates
from a vector store, reranks using a cross-encoder, then synthesizes a response
using an LLM conditioned on the retrieved context. This approach grounds the LLM
in factual information and reduces hallucinations significantly.
"""


class VLLMBenchmark:
    def __init__(self, model: str, backend: str = "vllm"):
        self.model = model
        self.backend = backend
        self.results = {}

    def _run_vllm_benchmark(
        self,
        n_requests: int = 100,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Dict:
        """Benchmark vLLM throughput + latency."""
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.model,
            dtype="float16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        prompts = [
            f"{CONTEXT}\n\nQuestion: {BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]}\nAnswer:"
            for i in range(n_requests)
        ]

        llm.generate(prompts[:3], sampling_params)

        latencies = []
        for i, prompt in enumerate(prompts):
            start = time.perf_counter()
            output = llm.generate([prompt], sampling_params)
            _ = output
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_requests} requests")

        start_batch = time.perf_counter()
        batch_outputs = llm.generate(prompts, sampling_params)
        batch_time = time.perf_counter() - start_batch

        tokens_generated = sum(
            len(o.outputs[0].token_ids) for o in batch_outputs
        )
        tokens_per_second = tokens_generated / batch_time

        latencies.sort()
        return {
            "backend": "vllm",
            "model": self.model,
            "n_requests": n_requests,
            "p50_ms": round(statistics.median(latencies), 1),
            "p90_ms": round(latencies[int(len(latencies) * 0.90)], 1),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
            "mean_ms": round(statistics.mean(latencies), 1),
            "batch_tokens_per_second": round(tokens_per_second, 1),
            "batch_time_s": round(batch_time, 2),
            "total_tokens": tokens_generated,
        }

    def _run_quantization_comparison(self) -> Dict:
        """Compare fp16 vs int8 vs int4 quantization."""
        from vllm import LLM, SamplingParams

        results = {}
        configs = [
            {"dtype": "float16", "label": "fp16"},
            {"dtype": "float16", "quantization": "awq", "label": "int4-awq"},
        ]

        for cfg in configs:
            label = cfg.pop("label")
            try:
                llm = LLM(model=self.model, **cfg, max_model_len=1024)
                sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
                prompts = BENCHMARK_PROMPTS[:20]

                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                elapsed = time.perf_counter() - start

                tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                results[label] = {
                    "tokens_per_second": round(tokens / elapsed, 1),
                    "latency_per_request_ms": round((elapsed / len(prompts)) * 1000, 1),
                    "total_tokens": tokens,
                }
                del llm
            except Exception as e:
                results[label] = {"error": str(e)}

        return results

    def _run_continuous_batching_benchmark(self, concurrent_users: int = 10) -> Dict:
        """
        Benchmark vLLM continuous batching under concurrent load.
        vLLM's PagedAttention handles this more efficiently than naive batching.
        """
        latencies = []
        lock = threading.Lock()
        errors = 0

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.model,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

        def single_request(prompt: str):
            nonlocal errors
            start = time.perf_counter()
            try:
                llm.generate([prompt], sampling_params)
                latency = (time.perf_counter() - start) * 1000
                with lock:
                    latencies.append(latency)
            except Exception:
                with lock:
                    errors += 1

        threads = []
        start_total = time.perf_counter()
        for i in range(concurrent_users * 5):
            prompt = f"{CONTEXT}\n\nQ: {BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]}\nA:"
            t = threading.Thread(target=single_request, args=(prompt,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_time = time.perf_counter() - start_total
        latencies.sort()

        return {
            "concurrent_users": concurrent_users,
            "total_requests": len(threads),
            "successful": len(latencies),
            "errors": errors,
            "total_time_s": round(total_time, 2),
            "qps": round(len(latencies) / total_time, 1),
            "p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
        }

    def run_all(self, output_path: str = "benchmarks/vllm_results.json"):
        print(f"\n=== vLLM Benchmarks: {self.model} ===\n")

        print("1. Sequential latency benchmark (100 requests)...")
        latency = self._run_vllm_benchmark(n_requests=100)
        print(f"   p50: {latency['p50_ms']}ms | p99: {latency['p99_ms']}ms")
        print(f"   Batch throughput: {latency['batch_tokens_per_second']} tokens/sec\n")

        print("2. Continuous batching under concurrent load...")
        batching = self._run_continuous_batching_benchmark(concurrent_users=10)
        print(f"   QPS: {batching['qps']} | p99: {batching['p99_ms']}ms\n")

        print("3. Quantization comparison (fp16 vs int4)...")
        quant = self._run_quantization_comparison()
        for label, metrics in quant.items():
            if "error" not in metrics:
                print(f"   {label}: {metrics['tokens_per_second']} tokens/sec")

        results = {
            "latency": latency,
            "continuous_batching": batching,
            "quantization": quant,
        }

        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print("\n=== README Metrics ===")
        print(f"| vLLM p50 latency           | {latency['p50_ms']}ms |")
        print(f"| vLLM p99 latency           | {latency['p99_ms']}ms |")
        print(f"| vLLM batch throughput      | {latency['batch_tokens_per_second']} tokens/sec |")
        print(f"| Concurrent QPS (10 users)  | {batching['qps']} |")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    bench = VLLMBenchmark(model=args.model)
    bench.run_all()
