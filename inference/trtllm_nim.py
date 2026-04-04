"""
TensorRT-LLM and NVIDIA NIM inference backends.

TensorRT-LLM:
  NVIDIA's LLM-specific inference stack — distinct from base TensorRT.
  Adds: continuous batching, in-flight batching, paged KV cache,
        tensor parallelism, speculative decoding, and FP8 quantization.

NVIDIA NIM:
  Managed inference microservice exposing OpenAI-compatible API.
  Runs on NVIDIA-optimised containers; drop-in replacement for OpenAI SDK.

Both backends share the same GenerationResult interface as vLLM and
llamacpp backends, enabling easy A/B switching in the inference pipeline.

Usage:
    # TensorRT-LLM (requires GPU + built engine)
    backend = TRTLLMBackend.from_hf("meta-llama/Llama-3.1-8B-Instruct")
    result  = backend.generate("Explain RAG in one sentence.")

    # NIM — local container
    nim = NIMBackend.local(port=8000, model="meta/llama-3.1-8b-instruct")
    result = nim.generate("Explain RAG in one sentence.")

    # NIM — NVIDIA cloud API
    nim = NIMBackend.cloud(model="meta/llama-3.1-8b-instruct")

    # Unified factory (same interface as vLLM / llamacpp backends)
    backend = InferenceBackendFactory.create("nim", model="meta/llama-3.1-8b-instruct")

    # Generate Docker Compose for NIM container
    python inference/trtllm_nim.py --gen-docker --model meta/llama-3.1-8b-instruct

    # Generate TRT-LLM build shell script
    python inference/trtllm_nim.py --gen-build-script --model meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Shared generation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Unified result dataclass shared across all inference backends
    (TRT-LLM, NIM, vLLM, llama.cpp).
    """
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    tokens_per_second: float
    backend: str
    model: str
    finish_reason: str = "stop"

    def __str__(self) -> str:
        return (
            f"[{self.backend}/{self.model}] {self.text[:200]}\n"
            f"  {self.prompt_tokens}+{self.completion_tokens} tokens | "
            f"{self.total_latency_ms:.1f}ms | {self.tokens_per_second:.1f} tok/s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT-LLM configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TRTLLMConfig:
    """Runtime config for a loaded TRT-LLM engine."""
    model_dir: str = "trtllm_engines"
    max_batch_size: int = 8
    max_input_len: int = 2048
    max_output_len: int = 512
    max_beam_width: int = 1
    # Parallelism
    tp_size: int = 1                # tensor parallelism degree
    pp_size: int = 1                # pipeline parallelism degree
    # Precision
    dtype: str = "float16"          # float16 | bfloat16 | float8
    quant_mode: str = "none"        # none | int8_kv_cache | fp8 | int4_awq | int4_gptq
    # KV cache
    kv_cache_free_gpu_memory_fraction: float = 0.9
    # Batching
    enable_chunked_prefill: bool = True
    enable_context_fmha: bool = True
    # Speculative decoding
    speculative_decoding_mode: str = "none"  # none | draft_tokens_external | medusa


@dataclass
class TRTLLMBuildConfig:
    """Config for building TRT-LLM engines from HuggingFace checkpoints."""
    hf_model_dir: str = ""
    output_dir: str = "trtllm_engines"
    tp_size: int = 1
    pp_size: int = 1
    dtype: str = "float16"
    quant_mode: str = "none"        # none | fp8 | int4_awq | int4_gptq
    max_batch_size: int = 8
    max_input_len: int = 2048
    max_output_len: int = 512
    use_gpt_attention_plugin: str = "float16"
    use_gemm_plugin: str = "float16"
    enable_paged_kv_cache: bool = True
    remove_input_padding: bool = True
    use_inflight_batching: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT-LLM engine builder
# ──────────────────────────────────────────────────────────────────────────────

class TRTLLMEngineBuilder:
    """
    Build TensorRT-LLM engines from HuggingFace checkpoints.

    Two-stage process:
      1. Convert HF checkpoint → TRT-LLM checkpoint format
         (handles weight layout, quantization calibration)
      2. Build TRT-LLM engine with trtllm-build CLI
         (fuses ops, selects optimal CUDA kernels, sets batch/seq limits)

    Hardware requirement: NVIDIA GPU with CUDA toolkit + tensorrt_llm package.
    The generated engine is specific to the GPU architecture it was built on.
    """

    def __init__(self, cfg: TRTLLMBuildConfig):
        self.cfg = cfg

    def build(self) -> str:
        """
        Run both conversion and build stages.
        Returns path to the engine directory.
        """
        import subprocess

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        checkpoint_dir = f"{self.cfg.output_dir}/checkpoint"
        engine_dir = f"{self.cfg.output_dir}/engine"

        # Stage 1: Convert HF checkpoint → TRT-LLM checkpoint
        logger.info("Stage 1: Converting HF checkpoint: %s", self.cfg.hf_model_dir)
        convert_cmd = [
            "python", "-m", "tensorrt_llm.commands.convert_checkpoint",
            "--model_dir", self.cfg.hf_model_dir,
            "--output_dir", checkpoint_dir,
            "--dtype", self.cfg.dtype,
            "--tp_size", str(self.cfg.tp_size),
            "--pp_size", str(self.cfg.pp_size),
        ]
        if self.cfg.quant_mode == "int4_awq":
            convert_cmd += ["--use_weight_only", "--weight_only_precision", "int4_awq"]
        elif self.cfg.quant_mode == "fp8":
            convert_cmd += ["--enable_fp8", "--fp8_kv_cache"]
        elif self.cfg.quant_mode == "int4_gptq":
            convert_cmd += ["--use_weight_only", "--weight_only_precision", "int4"]

        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"TRT-LLM checkpoint conversion failed:\n{result.stderr[:500]}")
        logger.info("Checkpoint converted: %s", checkpoint_dir)

        # Stage 2: Build engine
        logger.info("Stage 2: Building TRT-LLM engine (takes several minutes)…")
        build_cmd = [
            "trtllm-build",
            "--checkpoint_dir", checkpoint_dir,
            "--output_dir", engine_dir,
            "--max_batch_size", str(self.cfg.max_batch_size),
            "--max_input_len", str(self.cfg.max_input_len),
            "--max_output_len", str(self.cfg.max_output_len),
            "--gpt_attention_plugin", self.cfg.use_gpt_attention_plugin,
            "--gemm_plugin", self.cfg.use_gemm_plugin,
        ]
        if self.cfg.enable_paged_kv_cache:
            build_cmd.append("--paged_kv_cache")
        if self.cfg.remove_input_padding:
            build_cmd.append("--remove_input_padding")
        if self.cfg.use_inflight_batching:
            build_cmd.append("--use_inflight_batching")

        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"TRT-LLM engine build failed:\n{result.stderr[:500]}")

        logger.info("Engine built: %s", engine_dir)
        return engine_dir

    def generate_build_script(self, output_path: str = "scripts/build_trtllm.sh") -> str:
        """
        Write a self-contained shell script that reproduces the build.
        Useful for CI pipelines and documentation.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        quant_flags = ""
        if self.cfg.quant_mode == "int4_awq":
            quant_flags = "    --use_weight_only --weight_only_precision int4_awq \\\n"
        elif self.cfg.quant_mode == "fp8":
            quant_flags = "    --enable_fp8 --fp8_kv_cache \\\n"

        script = f"""#!/usr/bin/env bash
# TensorRT-LLM Engine Build Script
# Generated by LLMOps Research Assistant
# Requires: NVIDIA GPU, CUDA toolkit >= 11.8, tensorrt_llm package
# Install: pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com

set -euo pipefail

HF_MODEL_DIR="{self.cfg.hf_model_dir}"
OUTPUT_DIR="{self.cfg.output_dir}"
TP_SIZE={self.cfg.tp_size}
PP_SIZE={self.cfg.pp_size}
DTYPE="{self.cfg.dtype}"

echo "=== Stage 1: Converting HuggingFace checkpoint ==="
python -m tensorrt_llm.commands.convert_checkpoint \\
    --model_dir "$HF_MODEL_DIR" \\
    --output_dir "$OUTPUT_DIR/checkpoint" \\
    --dtype "$DTYPE" \\
    --tp_size "$TP_SIZE" \\
    --pp_size "$PP_SIZE" \\
{quant_flags}
echo "=== Stage 2: Building TRT-LLM engine ==="
trtllm-build \\
    --checkpoint_dir "$OUTPUT_DIR/checkpoint" \\
    --output_dir "$OUTPUT_DIR/engine" \\
    --max_batch_size {self.cfg.max_batch_size} \\
    --max_input_len {self.cfg.max_input_len} \\
    --max_output_len {self.cfg.max_output_len} \\
    --gpt_attention_plugin {self.cfg.use_gpt_attention_plugin} \\
    --gemm_plugin {self.cfg.use_gemm_plugin} \\
    --paged_kv_cache \\
    --remove_input_padding \\
    --use_inflight_batching

echo "Engine built at: $OUTPUT_DIR/engine"
"""
        with open(output_path, "w") as f:
            f.write(script)
        os.chmod(output_path, 0o755)
        logger.info("Build script written: %s", output_path)
        return output_path


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT-LLM runtime backend
# ──────────────────────────────────────────────────────────────────────────────

class TRTLLMBackend:
    """
    TensorRT-LLM inference backend with continuous batching.

    Uses the high-level tensorrt_llm.LLM API which handles:
      - In-flight batching (new requests join mid-batch)
      - Paged KV cache (memory-efficient attention)
      - Tensor parallelism across multiple GPUs

    Requires: tensorrt_llm package + a pre-built engine directory.
    Falls back to a mock response when the package is unavailable.
    """

    def __init__(self, cfg: TRTLLMConfig):
        self.cfg = cfg
        self._runner = None

    @classmethod
    def from_hf(
        cls,
        hf_model_id: str,
        engine_dir: Optional[str] = None,
        tp_size: int = 1,
    ) -> "TRTLLMBackend":
        """Convenience constructor — derives engine_dir from model ID."""
        engine_dir = engine_dir or f"trtllm_engines/{hf_model_id.replace('/', '_')}/engine"
        return cls(TRTLLMConfig(model_dir=engine_dir, tp_size=tp_size))

    def _load(self) -> None:
        try:
            from tensorrt_llm import LLM

            self._runner = LLM(
                model=self.cfg.model_dir,
                tensor_parallel_size=self.cfg.tp_size,
                pipeline_parallel_size=self.cfg.pp_size,
                dtype=self.cfg.dtype,
                kv_cache_config={
                    "free_gpu_memory_fraction": self.cfg.kv_cache_free_gpu_memory_fraction,
                },
                enable_chunked_prefill=self.cfg.enable_chunked_prefill,
            )
            logger.info(
                "TRT-LLM engine loaded: %s (tp=%d pp=%d)",
                self.cfg.model_dir, self.cfg.tp_size, self.cfg.pp_size,
            )
        except ImportError:
            logger.warning(
                "tensorrt_llm not installed — requires NVIDIA GPU + CUDA. "
                "Install: pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com"
            )
            self._runner = None

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> GenerationResult:
        if self._runner is None:
            self._load()
        if self._runner is None:
            return GenerationResult(
                text=f"[TRT-LLM unavailable — mock] {prompt[:60]}",
                prompt_tokens=0, completion_tokens=0,
                total_latency_ms=0.0, tokens_per_second=0.0,
                backend="trtllm", model=self.cfg.model_dir,
            )

        from tensorrt_llm import SamplingParams

        t0 = time.perf_counter()
        outputs = self._runner.generate(
            [prompt],
            sampling_params=SamplingParams(
                max_new_tokens=max_tokens or self.cfg.max_output_len,
                temperature=0.0,
                top_p=1.0,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = outputs[0].outputs[0].text
        completion_tokens = len(text.split())

        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=completion_tokens,
            total_latency_ms=latency_ms,
            tokens_per_second=completion_tokens / max(latency_ms / 1000, 1e-6),
            backend="trtllm",
            model=self.cfg.model_dir,
        )

    def generate_batch(
        self, prompts: List[str], max_tokens: Optional[int] = None
    ) -> List[GenerationResult]:
        """
        Continuous batching — submit all prompts as a single batch.
        TRT-LLM's in-flight batching scheduler handles variable-length
        sequences and interleaves prefill/decode steps for maximum GPU utilisation.
        """
        if self._runner is None:
            self._load()

        from tensorrt_llm import SamplingParams

        t0 = time.perf_counter()
        outputs = self._runner.generate(
            prompts,
            sampling_params=SamplingParams(
                max_new_tokens=max_tokens or self.cfg.max_output_len,
                temperature=0.0,
            ),
        )
        total_ms = (time.perf_counter() - t0) * 1000
        per_request_ms = total_ms / max(len(prompts), 1)

        return [
            GenerationResult(
                text=o.outputs[0].text,
                prompt_tokens=len(p.split()),
                completion_tokens=len(o.outputs[0].text.split()),
                total_latency_ms=per_request_ms,
                tokens_per_second=len(o.outputs[0].text.split()) / max(per_request_ms / 1000, 1e-6),
                backend="trtllm",
                model=self.cfg.model_dir,
            )
            for p, o in zip(prompts, outputs)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# NVIDIA NIM backend
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NIMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "meta/llama-3.1-8b-instruct"
    api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", "nim-api-key"))
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 30.0


class NIMBackend:
    """
    NVIDIA NIM inference backend.

    NIM exposes an OpenAI-compatible REST API, so this backend uses the
    standard openai SDK pointed at a NIM endpoint. No custom client needed.

    Deployment options:
      1. Local NIM container  — docker run --gpus all nvcr.io/nim/meta/llama-3.1-8b-instruct
      2. NVIDIA cloud NIM     — integrate.api.nvidia.com (managed, pay-per-token)
      3. Azure NIM            — Azure Marketplace NIM containers

    The NIM container internally uses TRT-LLM for optimised inference,
    so you get TRT-LLM performance without managing engine builds.
    """

    def __init__(self, cfg: Optional[NIMConfig] = None):
        self.cfg = cfg or NIMConfig()
        self._client = None

    @classmethod
    def local(cls, port: int = 8000, model: str = "meta/llama-3.1-8b-instruct") -> "NIMBackend":
        """Connect to a locally running NIM container."""
        return cls(NIMConfig(base_url=f"http://localhost:{port}/v1", model=model))

    @classmethod
    def cloud(cls, model: str = "meta/llama-3.1-8b-instruct") -> "NIMBackend":
        """Connect to NVIDIA's managed cloud NIM API."""
        return cls(NIMConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model=model,
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
        ))

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.cfg.base_url,
                    api_key=self.cfg.api_key,
                    timeout=self.cfg.timeout,
                )
            except ImportError:
                raise ImportError("openai required: pip install openai")
        return self._client

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> GenerationResult:
        client = self._get_client()
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        text = response.choices[0].message.content

        return GenerationResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_latency_ms=latency_ms,
            tokens_per_second=(usage.completion_tokens / max(latency_ms / 1000, 1e-6)) if usage else 0.0,
            backend="nim",
            model=self.cfg.model,
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    def stream(self, prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Yield token strings from a streaming NIM response."""
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def list_models(self) -> List[str]:
        """List models available at this NIM endpoint."""
        client = self._get_client()
        return [m.id for m in client.models.list().data]

    def health_check(self) -> Dict:
        """Check NIM container readiness."""
        try:
            import httpx
            base = self.cfg.base_url.rstrip("/").removesuffix("/v1")
            resp = httpx.get(f"{base}/v1/health/ready", timeout=5.0)
            return {
                "status": "ready" if resp.status_code == 200 else "not_ready",
                "code": resp.status_code,
                "endpoint": self.cfg.base_url,
            }
        except Exception as e:
            return {"status": "unreachable", "error": str(e), "endpoint": self.cfg.base_url}

    def benchmark(
        self,
        prompts: Optional[List[str]] = None,
        n_runs: int = 50,
    ) -> Dict:
        """Measure NIM endpoint latency and throughput."""
        import numpy as np

        if prompts is None:
            prompts = [
                "Explain retrieval-augmented generation in one paragraph.",
                "What are the trade-offs between FAISS and BM25?",
                "Describe the RLHF training pipeline.",
            ] * (n_runs // 3 + 1)

        results = [self.generate(p) for p in prompts[:n_runs]]
        latencies = [r.total_latency_ms for r in results]
        tps = [r.tokens_per_second for r in results]

        return {
            "backend": "nim",
            "model": self.cfg.model,
            "n_requests": len(results),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_tps": float(np.mean(tps)),
            "throughput_rps": float(len(results) / (sum(latencies) / 1000)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# NIM Docker Compose generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_nim_docker_compose(
    model: str = "meta/llama-3.1-8b-instruct",
    port: int = 8000,
    n_gpus: int = 1,
) -> str:
    """
    Generate a Docker Compose file for running a NIM container locally.
    Requires: NVIDIA GPU + NGC API key (export NGC_API_KEY=...).
    """
    model_tag = model.replace("/", "-")
    return f"""\
# NVIDIA NIM Docker Compose — {model}
# Requires: NVIDIA GPU, Docker with NVIDIA Container Toolkit
# Setup:
#   export NGC_API_KEY=<your-key>
#   docker login nvcr.io --username '$oauthtoken' --password $NGC_API_KEY
#   docker compose -f docker-compose.nim.yml up -d

services:
  nim-{model_tag}:
    image: nvcr.io/nim/{model}:latest
    ports:
      - "{port}:8000"
    environment:
      - NGC_API_KEY=${{NGC_API_KEY}}
      - NIM_MAX_MODEL_LEN=4096
      - NIM_TENSOR_PARALLELISM={n_gpus}
    volumes:
      - nim-cache:/opt/nim/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {n_gpus}
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/v1/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 120s

volumes:
  nim-cache:
"""


# ──────────────────────────────────────────────────────────────────────────────
# Unified backend factory
# ──────────────────────────────────────────────────────────────────────────────

class InferenceBackendFactory:
    """
    Create inference backends by name.

    All backends expose .generate(prompt) → GenerationResult
    and .stream(prompt) → Iterator[str], enabling easy A/B switching.

    Backend selection guide:
      nim       — local NIM container (best for production, no GPU on client)
      nim-cloud — NVIDIA cloud API (no local GPU required)
      trtllm    — direct TRT-LLM engine (lowest latency, requires built engine)
      vllm      — vLLM (flexible, good for development)
      llamacpp  — llama.cpp (CPU inference, GGUF models)
    """

    @staticmethod
    def create(backend: str, model: str, **kwargs) -> object:
        if backend == "nim":
            base_url = kwargs.get("base_url", "http://localhost:8000/v1")
            port = kwargs.get("port", 8000)
            return NIMBackend.local(port=port, model=model)
        elif backend == "nim-cloud":
            return NIMBackend.cloud(model=model)
        elif backend == "trtllm":
            engine_dir = kwargs.get("engine_dir", f"trtllm_engines/{model.replace('/', '_')}/engine")
            tp_size = kwargs.get("tp_size", 1)
            return TRTLLMBackend(TRTLLMConfig(model_dir=engine_dir, tp_size=tp_size))
        elif backend == "vllm":
            try:
                from vllm import LLM
                return LLM(model=model, **{k: v for k, v in kwargs.items()
                                           if k not in ("engine_dir", "port", "base_url")})
            except ImportError:
                raise ImportError("vllm required: pip install vllm")
        elif backend == "llamacpp":
            from inference.llamacpp_backend import LlamaCppRunner
            return LlamaCppRunner.from_gguf(model)
        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Choose from: {InferenceBackendFactory.available_backends()}"
            )

    @staticmethod
    def available_backends() -> List[str]:
        return ["nim", "nim-cloud", "trtllm", "vllm", "llamacpp"]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRT-LLM / NIM inference utilities")
    parser.add_argument("--backend", default="nim",
                        choices=["nim", "nim-cloud", "trtllm"])
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    parser.add_argument("--prompt", default="Explain retrieval-augmented generation.")
    parser.add_argument("--gen-docker", action="store_true",
                        help="Write docker-compose.nim.yml")
    parser.add_argument("--gen-build-script", action="store_true",
                        help="Write scripts/build_trtllm.sh")
    parser.add_argument("--health", action="store_true",
                        help="Check NIM endpoint health")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run NIM latency benchmark")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-gpus", type=int, default=1)
    args = parser.parse_args()

    if args.gen_docker:
        compose = generate_nim_docker_compose(args.model, port=args.port, n_gpus=args.n_gpus)
        out = "docker-compose.nim.yml"
        with open(out, "w") as f:
            f.write(compose)
        print(f"Written: {out}")

    elif args.gen_build_script:
        cfg = TRTLLMBuildConfig(hf_model_dir=args.model)
        builder = TRTLLMEngineBuilder(cfg)
        path = builder.generate_build_script()
        print(f"Written: {path}")

    elif args.health:
        nim = NIMBackend.local(port=args.port, model=args.model)
        import json
        print(json.dumps(nim.health_check(), indent=2))

    elif args.benchmark:
        nim = NIMBackend.local(port=args.port, model=args.model)
        import json
        result = nim.benchmark(n_runs=20)
        print(json.dumps(result, indent=2))

    else:
        backend = InferenceBackendFactory.create(args.backend, args.model, port=args.port)
        result = backend.generate(args.prompt)
        print(result)
