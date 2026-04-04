"""
TensorRT-LLM Engine Builder & NVIDIA NIM Backend Adapter
=========================================================
Two complementary components for production LLM inference:

1. TensorRT-LLM Engine Builder
   - Configures and builds optimised TRT-LLM engines for Llama/Mistral/Mixtral
   - Supports FP16, INT8 (SmoothQuant), INT4 (AWQ/GPTQ) precision
   - In-flight batching, paged KV-cache, tensor parallelism
   - Produces engine files deployable on Triton Inference Server

2. NVIDIA NIM Backend Adapter
   - Drop-in replacement for the OpenAI client using NVIDIA NIM endpoints
   - NIM exposes an OpenAI-compatible API — zero code changes to callers
   - Supports streaming, function calling, and embeddings
   - Integrates with the existing LangGraph synthesizer

Hardware note:
   Engine building requires an A100/H100 with the TensorRT-LLM toolkit.
   The NIM adapter runs against any NIM endpoint (cloud or local NGC container)
   and requires no GPU on the client side.

Usage:
    # Build a TRT-LLM engine (requires GPU + TRT-LLM toolkit)
    builder = TRTLLMEngineBuilder(TRTLLMConfig(model_name="llama-3.1-8b"))
    builder.build()

    # Use NIM as an OpenAI-compatible backend (no GPU required on client)
    nim = NIMBackend(NIMConfig(api_key="nvapi-..."))
    response = nim.chat([{"role": "user", "content": "What is RAG?"}])

    # Benchmark NIM vs OpenAI
    bench = NIMBenchmark(nim_backend=nim)
    bench.run(prompts=["..."] * 100)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT-LLM configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TRTLLMConfig:
    """
    Full configuration for a TensorRT-LLM engine build.
    Maps directly to trtllm-build CLI arguments.
    """
    # Model identity
    model_name: str = "llama-3.1-8b"
    model_type: str = "llama"           # llama | mistral | mixtral | falcon | gpt2 | gemma
    hf_model_dir: str = ""              # path to HuggingFace checkpoint
    engine_output_dir: str = "engines/trtllm"

    # Precision & quantization
    dtype: str = "float16"              # float16 | bfloat16 | float32
    quant_mode: str = "none"            # none | int8_sq | int4_awq | int4_gptq | fp8
    # SmoothQuant (INT8)
    smoothquant_alpha: float = 0.5
    # AWQ / GPTQ (INT4)
    group_size: int = 128
    has_zero_point: bool = False

    # Parallelism
    tp_size: int = 1                    # tensor parallel degree
    pp_size: int = 1                    # pipeline parallel degree
    world_size: int = 1                 # tp_size * pp_size

    # Sequence lengths
    max_input_len: int = 2048
    max_output_len: int = 512
    max_batch_size: int = 32
    max_beam_width: int = 1

    # KV-cache
    paged_kv_cache: bool = True
    tokens_per_block: int = 128
    max_num_tokens: int = 8192          # total tokens in flight (in-flight batching)

    # In-flight batching
    enable_chunked_prefill: bool = True
    max_num_sequences: int = 256

    # Plugin settings
    use_gpt_attention_plugin: str = "float16"
    use_gemm_plugin: str = "float16"
    use_rmsnorm_plugin: str = "float16"
    enable_context_fmha: bool = True    # fused MHA for prefill

    # Speculative decoding
    speculative_decoding_mode: str = "none"  # none | draft_tokens_external | medusa
    num_draft_tokens: int = 5

    def to_build_args(self) -> List[str]:
        """Convert config to trtllm-build CLI arguments."""
        args = [
            f"--model_dir={self.hf_model_dir}",
            f"--output_dir={self.engine_output_dir}/{self.model_name}",
            f"--dtype={self.dtype}",
            f"--tp_size={self.tp_size}",
            f"--pp_size={self.pp_size}",
            f"--max_input_len={self.max_input_len}",
            f"--max_output_len={self.max_output_len}",
            f"--max_batch_size={self.max_batch_size}",
            f"--max_beam_width={self.max_beam_width}",
            f"--tokens_per_block={self.tokens_per_block}",
            f"--max_num_tokens={self.max_num_tokens}",
            f"--gpt_attention_plugin={self.use_gpt_attention_plugin}",
            f"--gemm_plugin={self.use_gemm_plugin}",
        ]
        if self.paged_kv_cache:
            args.append("--paged_kv_cache=enable")
        if self.enable_context_fmha:
            args.append("--context_fmha=enable")
        if self.enable_chunked_prefill:
            args.append("--use_paged_context_fmha=enable")
        if self.quant_mode == "int8_sq":
            args += [
                "--use_smooth_quant",
                f"--per_channel",
                f"--per_token",
                f"--smoothquant_alpha={self.smoothquant_alpha}",
            ]
        elif self.quant_mode in ("int4_awq", "int4_gptq"):
            args += [
                "--use_weight_only",
                "--weight_only_precision=int4_awq" if self.quant_mode == "int4_awq" else "--weight_only_precision=int4",
                f"--group_size={self.group_size}",
            ]
        elif self.quant_mode == "fp8":
            args.append("--strongly_typed")
        return args

    def to_triton_config(self) -> Dict:
        """Generate Triton model repository config for this engine."""
        return {
            "name": self.model_name,
            "backend": "tensorrtllm",
            "max_batch_size": self.max_batch_size,
            "model_transaction_policy": {"decoupled": True},
            "dynamic_batching": {
                "preferred_batch_size": [1, 2, 4, 8, 16, 32],
                "max_queue_delay_microseconds": 1000,
            },
            "input": [
                {"name": "input_ids", "data_type": "TYPE_INT32", "dims": [-1]},
                {"name": "input_lengths", "data_type": "TYPE_INT32", "dims": [1]},
                {"name": "request_output_len", "data_type": "TYPE_INT32", "dims": [1]},
                {"name": "streaming", "data_type": "TYPE_BOOL", "dims": [1], "optional": True},
                {"name": "temperature", "data_type": "TYPE_FP32", "dims": [1], "optional": True},
                {"name": "top_p", "data_type": "TYPE_FP32", "dims": [1], "optional": True},
                {"name": "top_k", "data_type": "TYPE_INT32", "dims": [1], "optional": True},
            ],
            "output": [
                {"name": "output_ids", "data_type": "TYPE_INT32", "dims": [-1, -1]},
                {"name": "sequence_length", "data_type": "TYPE_INT32", "dims": [-1]},
                {"name": "cum_log_probs", "data_type": "TYPE_FP32", "dims": [-1], "optional": True},
            ],
            "parameters": {
                "engine_dir": {"string_value": f"{self.engine_output_dir}/{self.model_name}"},
                "max_tokens_in_paged_kvcache": {"string_value": str(self.max_num_tokens)},
                "kv_cache_free_gpu_mem_fraction": {"string_value": "0.9"},
                "executor_worker_path": {"string_value": "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"},
                "batching_strategy": {"string_value": "inflight_fused_batching"},
                "decoupled_mode": {"string_value": "true"},
            },
        }


class TRTLLMEngineBuilder:
    """
    Builds TensorRT-LLM engines from HuggingFace checkpoints.

    Pipeline:
      1. Convert HF weights → TRT-LLM checkpoint format
      2. Quantize (optional: SmoothQuant INT8, AWQ INT4, FP8)
      3. Build TRT engine with trtllm-build
      4. Generate Triton model repository config
      5. Validate engine with a test inference

    Requires: tensorrt-llm package + NVIDIA GPU with CUDA 12+
    """

    def __init__(self, config: TRTLLMConfig):
        self.config = config

    def convert_checkpoint(self) -> str:
        """
        Convert HuggingFace checkpoint to TRT-LLM format.
        Uses the convert_checkpoint.py script from the TRT-LLM examples.
        """
        import subprocess

        output_dir = Path(self.config.engine_output_dir) / "checkpoint" / self.config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", f"examples/{self.config.model_type}/convert_checkpoint.py",
            f"--model_dir={self.config.hf_model_dir}",
            f"--output_dir={output_dir}",
            f"--dtype={self.config.dtype}",
            f"--tp_size={self.config.tp_size}",
            f"--pp_size={self.config.pp_size}",
        ]

        if self.config.quant_mode == "int4_awq":
            cmd += ["--use_weight_only", "--weight_only_precision=int4_awq",
                    f"--group_size={self.config.group_size}"]
        elif self.config.quant_mode == "int8_sq":
            cmd += ["--smoothquant", f"--per_channel", "--per_token"]

        logger.info("Converting checkpoint: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Checkpoint conversion failed:\n{result.stderr}")

        logger.info("Checkpoint converted to: %s", output_dir)
        return str(output_dir)

    def build(self, checkpoint_dir: Optional[str] = None) -> str:
        """
        Build TRT-LLM engine from checkpoint.
        Returns path to the built engine directory.
        """
        import subprocess

        if checkpoint_dir is None:
            checkpoint_dir = self.convert_checkpoint()

        engine_dir = Path(self.config.engine_output_dir) / self.config.model_name
        engine_dir.mkdir(parents=True, exist_ok=True)

        build_args = self.config.to_build_args()
        # Override model_dir to use converted checkpoint
        build_args = [a for a in build_args if not a.startswith("--model_dir")]
        build_args.insert(0, f"--checkpoint_dir={checkpoint_dir}")

        cmd = ["trtllm-build"] + build_args
        logger.info("Building TRT-LLM engine: %s", " ".join(cmd[:5]) + " ...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Engine build failed:\n{result.stderr}")

        logger.info("Engine built: %s", engine_dir)
        self._write_triton_config(engine_dir)
        return str(engine_dir)

    def _write_triton_config(self, engine_dir: Path) -> None:
        """Write config.pbtxt for Triton model repository."""
        triton_cfg = self.config.to_triton_config()
        config_path = engine_dir / "config.pbtxt"

        lines = [f'name: "{triton_cfg["name"]}"', f'backend: "{triton_cfg["backend"]}"',
                 f'max_batch_size: {triton_cfg["max_batch_size"]}']
        for key, val in triton_cfg.get("parameters", {}).items():
            lines.append(f'parameters {{ key: "{key}" value {{ string_value: "{val["string_value"]}" }} }}')

        config_path.write_text("\n".join(lines))
        logger.info("Triton config written: %s", config_path)

        # Also write JSON for inspection
        (engine_dir / "build_config.json").write_text(
            json.dumps(triton_cfg, indent=2)
        )

    def validate(self, engine_dir: str, prompt: str = "What is RAG?") -> Dict:
        """Run a test inference to validate the built engine."""
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner

            runner = ModelRunner.from_dir(engine_dir, rank=0)
            import torch
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)  # placeholder
            t0 = time.perf_counter()
            outputs = runner.generate(input_ids, max_new_tokens=50)
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("Engine validation passed. Latency: %.1f ms", latency_ms)
            return {"status": "ok", "latency_ms": latency_ms}
        except ImportError:
            logger.warning("tensorrt-llm not installed — skipping validation")
            return {"status": "skipped", "reason": "tensorrt-llm not installed"}
        except Exception as e:
            logger.error("Engine validation failed: %s", e)
            return {"status": "error", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# NVIDIA NIM backend adapter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NIMConfig:
    """
    Configuration for NVIDIA NIM inference endpoints.
    NIM exposes an OpenAI-compatible API — swap base_url to point at
    a self-hosted NIM container or the NVIDIA cloud API.
    """
    # NVIDIA cloud API
    api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    base_url: str = "https://integrate.api.nvidia.com/v1"

    # Model selection — see https://build.nvidia.com/explore/discover
    model: str = "meta/llama-3.1-8b-instruct"

    # Generation defaults
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.9
    stream: bool = False

    # Embeddings
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_truncate: str = "END"     # NONE | START | END

    # Timeout / retry
    timeout: float = 60.0
    max_retries: int = 3


class NIMBackend:
    """
    NVIDIA NIM inference backend.

    Drop-in replacement for the OpenAI client — uses the same API surface
    but routes to NVIDIA's optimised NIM endpoints (TRT-LLM under the hood).

    Integrates with the existing LangGraph synthesizer:
        synthesizer = SynthesizerAgent(llm_backend=NIMBackend(NIMConfig()))
    """

    def __init__(self, config: Optional[NIMConfig] = None):
        self.config = config or NIMConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
                logger.info("NIM client connected: %s → %s",
                            self.config.base_url, self.config.model)
            except ImportError:
                raise ImportError("openai>=1.0 required: pip install openai")
        return self._client

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Send a chat completion request to NIM.
        Returns the response string (or a streaming iterator if stream=True).
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            stream=stream if stream is not None else self.config.stream,
        )
        if stream or self.config.stream:
            return response  # caller iterates chunks
        return response.choices[0].message.content

    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """Yield token strings from a streaming NIM response."""
        for chunk in self.chat(messages, stream=True, **kwargs):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "query",   # "query" | "passage"
    ) -> List[List[float]]:
        """
        Get embeddings from NIM's embedding endpoint.
        Uses nvidia/nv-embedqa-e5-v5 by default (optimised for RAG).
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=model or self.config.embedding_model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": self.config.embedding_truncate,
            },
        )
        return [item.embedding for item in response.data]

    def list_models(self) -> List[str]:
        """List available NIM models."""
        client = self._get_client()
        return [m.id for m in client.models.list().data]

    def health_check(self) -> Dict:
        """Verify NIM endpoint is reachable and responding."""
        try:
            t0 = time.perf_counter()
            response = self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return {"status": "ok", "latency_ms": latency_ms, "model": self.config.model}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# NIM benchmarking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NIMBenchResult:
    model: str
    n_requests: int
    total_time_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    errors: int

    def __str__(self) -> str:
        return (
            f"NIM Benchmark — {self.model}\n"
            f"  Requests: {self.n_requests}  Errors: {self.errors}\n"
            f"  Latency  mean={self.mean_latency_ms:.0f}ms  "
            f"p50={self.p50_latency_ms:.0f}ms  "
            f"p95={self.p95_latency_ms:.0f}ms  "
            f"p99={self.p99_latency_ms:.0f}ms\n"
            f"  Throughput: {self.throughput_rps:.1f} req/s"
        )


class NIMBenchmark:
    """Benchmark NIM endpoint latency and throughput."""

    def __init__(self, nim_backend: NIMBackend):
        self.nim = nim_backend

    def run(
        self,
        prompts: Optional[List[str]] = None,
        n_requests: int = 50,
        max_tokens: int = 100,
    ) -> NIMBenchResult:
        import statistics

        if prompts is None:
            prompts = [
                "Explain retrieval-augmented generation in one paragraph.",
                "What are the trade-offs between FAISS and BM25 retrieval?",
                "Describe the RLHF training pipeline.",
            ] * (n_requests // 3 + 1)

        prompts = prompts[:n_requests]
        latencies: List[float] = []
        errors = 0

        logger.info("Running NIM benchmark: %d requests, model=%s", n_requests, self.nim.config.model)
        t_total_start = time.perf_counter()

        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            try:
                self.nim.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                errors += 1
                logger.warning("Request %d failed: %s", i, e)

        total_time_s = time.perf_counter() - t_total_start

        if not latencies:
            raise RuntimeError("All requests failed")

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        result = NIMBenchResult(
            model=self.nim.config.model,
            n_requests=n_requests,
            total_time_s=total_time_s,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies_sorted[int(n * 0.50)],
            p95_latency_ms=latencies_sorted[int(n * 0.95)],
            p99_latency_ms=latencies_sorted[min(int(n * 0.99), n - 1)],
            throughput_rps=len(latencies) / total_time_s,
            errors=errors,
        )
        logger.info("\n%s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT-LLM / NIM utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Build engine
    build_p = subparsers.add_parser("build", help="Build TRT-LLM engine")
    build_p.add_argument("--model-name", default="llama-3.1-8b")
    build_p.add_argument("--model-dir", required=True)
    build_p.add_argument("--output-dir", default="engines/trtllm")
    build_p.add_argument("--dtype", default="float16")
    build_p.add_argument("--quant", default="none",
                         choices=["none", "int8_sq", "int4_awq", "int4_gptq", "fp8"])
    build_p.add_argument("--tp-size", type=int, default=1)

    # NIM health check
    nim_p = subparsers.add_parser("nim-health", help="Check NIM endpoint health")
    nim_p.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    nim_p.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")

    # NIM benchmark
    bench_p = subparsers.add_parser("nim-bench", help="Benchmark NIM endpoint")
    bench_p.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    bench_p.add_argument("--n-requests", type=int, default=20)

    args = parser.parse_args()

    if args.command == "build":
        cfg = TRTLLMConfig(
            model_name=args.model_name,
            hf_model_dir=args.model_dir,
            engine_output_dir=args.output_dir,
            dtype=args.dtype,
            quant_mode=args.quant,
            tp_size=args.tp_size,
        )
        builder = TRTLLMEngineBuilder(cfg)
        engine_dir = builder.build()
        print(f"Engine built: {engine_dir}")
        result = builder.validate(engine_dir)
        print(f"Validation: {result}")

    elif args.command == "nim-health":
        cfg = NIMConfig(model=args.model, base_url=args.base_url)
        nim = NIMBackend(cfg)
        print(json.dumps(nim.health_check(), indent=2))

    elif args.command == "nim-bench":
        cfg = NIMConfig(model=args.model)
        nim = NIMBackend(cfg)
        bench = NIMBenchmark(nim)
        bench.run(n_requests=args.n_requests)

    else:
        parser.print_help()
