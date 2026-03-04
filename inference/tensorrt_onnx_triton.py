"""
TensorRT / ONNX / Triton Inference Server — Production Inference Stack
=======================================================================
Production inference optimization pipeline:

  1. ONNX export       — convert HuggingFace model to ONNX with dynamic axes
  2. TensorRT engine   — compile ONNX → TRT engine (FP32 / FP16 / INT8 / FP8)
  3. Triton serving    — generate model repository + config.pbtxt + ensemble
  4. Benchmarking      — latency / throughput profiling across backends
  5. Triton client     — HTTP inference client with embedding support

Backends compared:
  PyTorch (baseline) → ONNX Runtime → TensorRT FP16 → TensorRT INT8-PTQ

Usage:
    pipeline = InferenceOptimizationPipeline.from_hf("meta-llama/Llama-3.2-1B")
    pipeline.export_onnx("artifacts/model.onnx")
    pipeline.build_trt_engine("artifacts/model.trt", precision="fp16")
    pipeline.generate_triton_config("triton_repo/", backend="tensorrt")

    # Or run the full pipeline end-to-end:
    python inference/tensorrt_onnx_triton.py --model meta-llama/Llama-3.2-1B --precision fp16
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len: int = 512
    max_batch_size: int = 8
    precision: str = "fp16"           # fp32 | fp16 | int8 | fp8
    workspace_gb: int = 8             # TensorRT builder workspace
    calibration_samples: int = 512    # for INT8 PTQ calibration
    triton_model_name: str = "llmops-llm"
    triton_max_batch: int = 8
    triton_url: str = "localhost:8000"
    onnx_opset: int = 17
    artifacts_dir: str = "artifacts/trt"


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────────────────────────────────────

class ONNXExporter:
    """
    Export HuggingFace transformer models to ONNX with dynamic axes.
    Supports encoder-only, decoder-only, and encoder-decoder architectures.
    """

    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg

    def export(
        self,
        model,
        tokenizer,
        output_path: str,
        opset: Optional[int] = None,
    ) -> str:
        import torch

        opset = opset or self.cfg.onnx_opset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        dummy_text = "Explain retrieval-augmented generation in one sentence."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=self.cfg.max_seq_len,
            padding="max_length",
            truncation=True,
        )

        model.eval()
        input_names = list(inputs.keys())
        output_names = ["logits"]

        dynamic_axes = {name: {0: "batch_size", 1: "sequence_length"}
                        for name in input_names}
        dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}

        logger.info("Exporting to ONNX (opset=%d): %s", opset, output_path)
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs[k] for k in input_names),
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

        self._verify_onnx(output_path)
        file_mb = os.path.getsize(output_path) / (1024 ** 2)
        logger.info("ONNX export complete: %.1f MB at %s", file_mb, output_path)
        return output_path

    def _verify_onnx(self, path: str) -> None:
        try:
            import onnx
            model = onnx.load(path)
            onnx.checker.check_model(model)
            logger.info("ONNX model check passed (opset %d).", model.opset_import[0].version)
        except ImportError:
            logger.warning("onnx not installed — skipping model check.")

    def optimize_onnx(self, input_path: str, output_path: str) -> str:
        """Apply ONNX graph optimizations (constant folding, node fusion)."""
        try:
            from onnxruntime.transformers import optimizer as ort_optimizer

            optimized = ort_optimizer.optimize_model(
                input_path,
                model_type="bert",
                num_heads=12,
                hidden_size=768,
                opt_level=99,
                use_gpu=True,
            )
            optimized.save_model_to_file(output_path)
            logger.info("ONNX graph optimized: %s", output_path)
            return output_path
        except Exception as e:
            logger.warning("ONNX optimization skipped: %s", e)
            return input_path


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime backend
# ──────────────────────────────────────────────────────────────────────────────

class ONNXRuntimeBackend:
    """ONNX Runtime inference backend (CPU + CUDA execution providers)."""

    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._session = None

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.use_gpu else ["CPUExecutionProvider"]
            )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=providers
            )
            logger.info("ORT session loaded (providers=%s)", providers)
        return self._session

    def run(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        session = self._get_session()
        output_names = [o.name for o in session.get_outputs()]
        return session.run(output_names, inputs)[0]

    def benchmark(self, inputs: Dict[str, np.ndarray], n_runs: int = 100) -> Dict:
        for _ in range(10):
            self.run(inputs)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(inputs)
            latencies.append((time.perf_counter() - t0) * 1000)
        return _latency_stats(latencies, backend="onnxruntime")


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT engine builder
# ──────────────────────────────────────────────────────────────────────────────

class TensorRTEngineBuilder:
    """
    Build TensorRT engines from ONNX models.
    Supports FP32, FP16, INT8 (PTQ with calibration), and FP8.
    """

    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg

    def build_engine(
        self,
        onnx_path: str,
        engine_path: str,
        precision: Optional[str] = None,
        calibrator=None,
    ) -> str:
        precision = precision or self.cfg.precision
        try:
            import tensorrt as trt
        except ImportError:
            logger.warning("TensorRT not installed — writing stub config instead.")
            return self._mock_engine_config(engine_path, precision)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config = builder.create_builder_config()

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.cfg.workspace_gb * (1 << 30),
        )

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT: FP16 precision enabled.")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
                logger.info("TRT: INT8 PTQ calibration enabled.")
            else:
                logger.warning("INT8 requested but no calibrator — falling back to FP16.")
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("TRT: FP8 precision enabled.")

        # Dynamic shapes
        profile = builder.create_optimization_profile()
        for tensor_name in ("input_ids", "attention_mask"):
            profile.set_shape(
                tensor_name,
                min=(1, 1),
                opt=(self.cfg.max_batch_size // 2, self.cfg.max_seq_len // 2),
                max=(self.cfg.max_batch_size, self.cfg.max_seq_len),
            )
        config.add_optimization_profile(profile)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TRT parse error: %s", parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        logger.info("Building TRT engine (precision=%s) — this may take several minutes…", precision)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed")

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

        size_mb = os.path.getsize(engine_path) / (1024 ** 2)
        logger.info("TRT engine built: %.1f MB at %s", size_mb, engine_path)
        return engine_path

    def _mock_engine_config(self, engine_path: str, precision: str) -> str:
        """Write a stub config when TRT is not installed (for CI/testing)."""
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        stub = {
            "engine_path": engine_path,
            "precision": precision,
            "max_batch_size": self.cfg.max_batch_size,
            "max_seq_len": self.cfg.max_seq_len,
            "workspace_gb": self.cfg.workspace_gb,
            "status": "mock — TensorRT not installed",
        }
        stub_path = engine_path + ".config.json"
        Path(stub_path).write_text(json.dumps(stub, indent=2))
        logger.info("TRT stub config written to %s", stub_path)
        return stub_path


# ──────────────────────────────────────────────────────────────────────────────
# INT8 PTQ Calibrator
# ──────────────────────────────────────────────────────────────────────────────

class INT8Calibrator:
    """
    Post-Training Quantization (PTQ) calibrator for TensorRT INT8.
    Implements IInt8EntropyCalibrator2 — feeds a representative calibration
    dataset to TensorRT to compute optimal INT8 activation scale factors.
    """

    def __init__(
        self,
        calibration_data: List[Dict[str, np.ndarray]],
        cache_file: str = "outputs/trt/calibration.cache",
    ):
        self.data = calibration_data
        self.cache_file = cache_file
        self._idx = 0

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: List[str]):
        try:
            import pycuda.driver as cuda
        except ImportError:
            return None

        if self._idx >= len(self.data):
            return None

        batch = self.data[self._idx]
        self._idx += 1
        buffers = []
        for name in names:
            arr = batch[name].astype(np.int32)
            buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(buf, arr)
            buffers.append(int(buf))
        return buffers

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info("Calibration cache written: %s", self.cache_file)


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT runtime
# ──────────────────────────────────────────────────────────────────────────────

class TensorRTRuntime:
    """Run inference on a compiled TensorRT engine."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self._engine = None
        self._context = None

    def _load(self) -> None:
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            with open(self.engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            logger.info("TRT engine loaded from %s", self.engine_path)
        except ImportError:
            logger.warning("TensorRT/PyCUDA not available — inference skipped.")

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if self._engine is None:
            self._load()
        if self._engine is None:
            raise RuntimeError("TRT engine not loaded.")

        import pycuda.driver as cuda

        batch, seq_len = input_ids.shape
        self._context.set_input_shape("input_ids", (batch, seq_len))
        self._context.set_input_shape("attention_mask", (batch, seq_len))

        d_input_ids = cuda.mem_alloc(input_ids.astype(np.int32).nbytes)
        d_attention_mask = cuda.mem_alloc(attention_mask.astype(np.int32).nbytes)
        out_shape = self._context.get_tensor_shape("logits")
        output = np.empty(out_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input_ids, input_ids.astype(np.int32), stream)
        cuda.memcpy_htod_async(d_attention_mask, attention_mask.astype(np.int32), stream)
        self._context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        return output

    def benchmark(self, inputs: Dict, n_runs: int = 100) -> Dict:
        latencies = []
        for _ in range(10):
            try:
                self.infer(inputs["input_ids"], inputs["attention_mask"])
            except Exception:
                break
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                self.infer(inputs["input_ids"], inputs["attention_mask"])
            except Exception:
                break
            latencies.append((time.perf_counter() - t0) * 1000)
        return _latency_stats(latencies, backend="tensorrt")


# ──────────────────────────────────────────────────────────────────────────────
# Triton config generator
# ──────────────────────────────────────────────────────────────────────────────

class TritonConfigGenerator:
    """
    Generates NVIDIA Triton Inference Server model repository structure.

    Output layout:
      triton_repo/
        llmops-llm/
          config.pbtxt                  ← model config
          1/
            model.plan                  ← TRT engine (or model.onnx for ORT)
        llmops-llm-ensemble/
          config.pbtxt                  ← ensemble pipeline config
          1/
    """

    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg

    def generate(
        self,
        repo_dir: str,
        backend: str = "tensorrt",      # "tensorrt" | "onnxruntime" | "pytorch"
        model_file: Optional[str] = None,
    ) -> str:
        model_dir = Path(repo_dir) / self.cfg.triton_model_name / "1"
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = Path(repo_dir) / self.cfg.triton_model_name / "config.pbtxt"
        config_path.write_text(self._build_config(backend))
        logger.info("Triton config written: %s", config_path)

        if model_file and os.path.exists(model_file):
            import shutil
            ext = ".plan" if backend == "tensorrt" else ".onnx"
            dest = model_dir / f"model{ext}"
            shutil.copy2(model_file, dest)
            logger.info("Model artifact copied to %s", dest)

        self._write_ensemble_config(repo_dir)
        self._write_deploy_instructions(Path(repo_dir))
        return str(config_path)

    def _build_config(self, backend: str) -> str:
        backend_map = {
            "tensorrt": "tensorrt_plan",
            "onnxruntime": "onnxruntime_onnx",
            "pytorch": "pytorch_libtorch",
        }
        triton_backend = backend_map.get(backend, "tensorrt_plan")

        return f"""name: "{self.cfg.triton_model_name}"
backend: "{triton_backend}"
max_batch_size: {self.cfg.triton_max_batch}

dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 5000
}}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator: [
      {{
        name: "{'tensorrt' if backend == 'tensorrt' else 'default'}"
        parameters {{
          key: "precision_mode"
          value: "{self.cfg.precision.upper()}"
        }}
        parameters {{
          key: "max_workspace_size_bytes"
          value: "{self.cfg.workspace_gb * (1 << 30)}"
        }}
      }}
    ]
  }}
}}

model_warmup [
  {{
    name: "warmup_batch1"
    batch_size: 1
    inputs {{
      key: "input_ids"
      value {{
        data_type: TYPE_INT64
        dims: [ 32 ]
        zero_data: true
      }}
    }}
  }}
]
"""

    def _write_ensemble_config(self, repo_dir: str) -> None:
        """Write a Triton ensemble config for tokenizer → LLM → detokenizer pipeline."""
        ensemble_dir = Path(repo_dir) / f"{self.cfg.triton_model_name}-ensemble" / "1"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        config = f"""name: "{self.cfg.triton_model_name}-ensemble"
platform: "ensemble"
max_batch_size: {self.cfg.triton_max_batch}

input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "{self.cfg.triton_model_name}"
      model_version: 1
      input_map {{
        key: "input_ids"
        value: "input_ids"
      }}
      output_map {{
        key: "logits"
        value: "logits"
      }}
    }}
  ]
}}
"""
        config_path = Path(repo_dir) / f"{self.cfg.triton_model_name}-ensemble" / "config.pbtxt"
        config_path.write_text(config)
        logger.info("Triton ensemble config written: %s", config_path)

    def _write_deploy_instructions(self, repo_root: Path) -> None:
        instructions = f"""# Triton Inference Server Deployment

## Start the server

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v {repo_root.absolute()}:/models \\
  nvcr.io/nvidia/tritonserver:24.01-py3 \\
  tritonserver --model-repository=/models
```

## Check server health

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/{self.cfg.triton_model_name}
```

## Run inference via TritonClient

```python
from inference.tensorrt_onnx_triton import TritonClient
client = TritonClient(url="{self.cfg.triton_url}", model_name="{self.cfg.triton_model_name}")
result = client.infer_raw(input_ids, attention_mask)
```
"""
        (repo_root / "DEPLOY.md").write_text(instructions)


# ──────────────────────────────────────────────────────────────────────────────
# Triton HTTP client
# ──────────────────────────────────────────────────────────────────────────────

class TritonClient:
    """
    HTTP client for Triton Inference Server.
    Supports raw tensor inference, embedding extraction, and benchmarking.
    Requires: pip install tritonclient[http]
    """

    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "llmops-llm",
        model_version: str = "1",
    ):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import tritonclient.http as httpclient
                self._client = httpclient.InferenceServerClient(url=self.url)
            except ImportError:
                raise ImportError("Install tritonclient: pip install tritonclient[http]")
        return self._client

    def is_ready(self) -> bool:
        try:
            return self._get_client().is_server_ready()
        except Exception as e:
            logger.warning("Triton server not ready: %s", e)
            return False

    def infer_raw(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        output_name: str = "logits",
    ) -> np.ndarray:
        """Send a batch to Triton and return the output tensor."""
        import tritonclient.http as httpclient

        client = self._get_client()
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids.astype(np.int64))
        inputs[1].set_data_from_numpy(attention_mask.astype(np.int64))

        outputs = [httpclient.InferRequestedOutput(output_name)]
        response = client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=self.model_version,
        )
        return response.as_numpy(output_name)

    def embed(
        self,
        texts: List[str],
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
    ) -> np.ndarray:
        """
        Tokenize texts locally, run through Triton, mean-pool to get embeddings.
        Returns (batch, hidden_size) float32 array.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        raw = self.infer_raw(
            encoded["input_ids"].astype(np.int64),
            encoded["attention_mask"].astype(np.int64),
            output_name="logits",
        )
        mask = encoded["attention_mask"][..., np.newaxis].astype(np.float32)
        return (raw * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

    def benchmark(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        n_runs: int = 100,
    ) -> Dict:
        for _ in range(5):
            self.infer_raw(input_ids, attention_mask)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.infer_raw(input_ids, attention_mask)
            latencies.append((time.perf_counter() - t0) * 1000)
        return _latency_stats(latencies, backend="triton-http")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    backend: str
    precision: str
    batch_size: int
    seq_len: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_qps: float
    memory_mb: float
    speedup_vs_pytorch: float = 1.0

    def __str__(self) -> str:
        return (
            f"{self.backend:20s} | prec={self.precision:5s} | "
            f"p50={self.p50_ms:6.1f}ms | p90={self.p90_ms:6.1f}ms | "
            f"p99={self.p99_ms:6.1f}ms | QPS={self.throughput_qps:6.1f} | "
            f"speedup={self.speedup_vs_pytorch:.2f}x"
        )


def _latency_stats(latencies: List[float], backend: str) -> Dict:
    if not latencies:
        return {"backend": backend, "p50": 0, "p90": 0, "p99": 0, "qps": 0, "n_runs": 0}
    arr = np.array(latencies)
    return {
        "backend": backend,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "qps": float(1000 / arr.mean()),
        "n_runs": len(latencies),
    }


class InferenceBenchmarker:
    """
    Side-by-side benchmarking: PyTorch → ONNX Runtime → TensorRT FP16 → TensorRT INT8.
    Measures p50/p90/p99 latency, throughput (QPS), and GPU memory.
    Logs results to MLflow.
    """

    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg
        self.results: List[BenchmarkResult] = []

    def benchmark_pytorch(self, model, inputs: Dict, n_runs: int = 200) -> BenchmarkResult:
        import torch

        model.eval()
        tensor_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        with torch.no_grad():
            for _ in range(20):
                model(**tensor_inputs)
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(**tensor_inputs)
                latencies.append((time.perf_counter() - t0) * 1000)
        stats = _latency_stats(latencies, "pytorch")
        result = BenchmarkResult(
            backend="pytorch", precision="fp32",
            batch_size=inputs["input_ids"].shape[0],
            seq_len=inputs["input_ids"].shape[1],
            p50_ms=stats["p50"], p90_ms=stats["p90"], p99_ms=stats["p99"],
            throughput_qps=stats["qps"],
            memory_mb=self._gpu_memory_mb(),
        )
        self.results.append(result)
        return result

    def benchmark_ort(
        self, ort_backend: ONNXRuntimeBackend, inputs: Dict, n_runs: int = 200
    ) -> BenchmarkResult:
        np_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        stats = ort_backend.benchmark(np_inputs, n_runs=n_runs)
        result = BenchmarkResult(
            backend="onnxruntime", precision="fp32",
            batch_size=inputs["input_ids"].shape[0],
            seq_len=inputs["input_ids"].shape[1],
            p50_ms=stats["p50"], p90_ms=stats["p90"], p99_ms=stats["p99"],
            throughput_qps=stats["qps"],
            memory_mb=self._gpu_memory_mb(),
        )
        self.results.append(result)
        return result

    def benchmark_trt(
        self, trt_runtime: TensorRTRuntime, inputs: Dict, precision: str = "fp16", n_runs: int = 200
    ) -> BenchmarkResult:
        stats = trt_runtime.benchmark(inputs, n_runs=n_runs)
        result = BenchmarkResult(
            backend="tensorrt", precision=precision,
            batch_size=inputs["input_ids"].shape[0],
            seq_len=inputs["input_ids"].shape[1],
            p50_ms=stats["p50"], p90_ms=stats["p90"], p99_ms=stats["p99"],
            throughput_qps=stats["qps"],
            memory_mb=self._gpu_memory_mb(),
        )
        self.results.append(result)
        return result

    def _gpu_memory_mb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception:
            pass
        return 0.0

    def print_summary(self) -> None:
        if not self.results:
            print("No benchmark results.")
            return
        baseline = self.results[0].p50_ms if self.results else 1.0
        print("\n" + "=" * 90)
        print(f"{'Backend':20s} | {'Prec':5s} | {'p50':>8s} | {'p90':>8s} | {'p99':>8s} | {'QPS':>8s} | {'Speedup':>8s}")
        print("=" * 90)
        for r in self.results:
            r.speedup_vs_pytorch = baseline / r.p50_ms if r.p50_ms > 0 else 1.0
            print(r)
        print("=" * 90)

    def log_to_mlflow(self) -> None:
        try:
            import mlflow
            with mlflow.start_run(run_name="inference-benchmark"):
                mlflow.log_param("model", self.cfg.model_name)
                for r in self.results:
                    prefix = r.backend.replace(" ", "_")
                    mlflow.log_metric(f"{prefix}_p50_ms", r.p50_ms)
                    mlflow.log_metric(f"{prefix}_p99_ms", r.p99_ms)
                    mlflow.log_metric(f"{prefix}_qps", r.throughput_qps)
                    mlflow.log_metric(f"{prefix}_speedup", r.speedup_vs_pytorch)
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

class InferenceOptimizationPipeline:
    """
    End-to-end inference optimization:
      load model → ONNX export → TRT engine → Triton config → benchmark

    Usage:
        pipeline = InferenceOptimizationPipeline.from_hf("meta-llama/Llama-3.2-1B")
        pipeline.run_full_pipeline()
    """

    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg
        self.exporter = ONNXExporter(cfg)
        self.trt_builder = TensorRTEngineBuilder(cfg)
        self.triton_gen = TritonConfigGenerator(cfg)
        self.benchmarker = InferenceBenchmarker(cfg)

    @classmethod
    def from_hf(cls, model_name: str, **kwargs) -> "InferenceOptimizationPipeline":
        cfg = OptimizationConfig(model_name=model_name, **kwargs)
        return cls(cfg)

    def load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s", self.cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        return model, tokenizer

    def export_onnx(
        self, output_path: str, model=None, tokenizer=None
    ) -> str:
        if model is None or tokenizer is None:
            model, tokenizer = self.load_model_and_tokenizer()
        return self.exporter.export(model, tokenizer, output_path)

    def build_trt_engine(
        self,
        engine_path: str,
        onnx_path: Optional[str] = None,
        precision: Optional[str] = None,
        calibrator=None,
    ) -> str:
        if onnx_path is None:
            onnx_path = engine_path.replace(".trt", ".onnx")
        return self.trt_builder.build_engine(onnx_path, engine_path, precision, calibrator)

    def generate_triton_config(
        self,
        repo_dir: str,
        backend: str = "tensorrt",
        model_file: Optional[str] = None,
    ) -> str:
        return self.triton_gen.generate(repo_dir, backend, model_file)

    def run_full_pipeline(self, triton_repo: str = "triton_repo") -> Dict[str, str]:
        """
        Run the complete pipeline:
          1. Load model
          2. Export to ONNX
          3. Build TensorRT engine
          4. Generate Triton model repository
        """
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)
        onnx_path = f"{self.cfg.artifacts_dir}/model.onnx"
        engine_path = f"{self.cfg.artifacts_dir}/model.trt"

        model, tokenizer = self.load_model_and_tokenizer()
        self.export_onnx(onnx_path, model, tokenizer)
        self.build_trt_engine(engine_path, onnx_path)
        config_path = self.generate_triton_config(
            triton_repo, backend="tensorrt", model_file=engine_path
        )

        artefacts = {
            "onnx": onnx_path,
            "trt_engine": engine_path,
            "triton_repo": triton_repo,
            "triton_config": config_path,
        }
        logger.info("Full pipeline complete: %s", artefacts)
        return artefacts


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRT/ONNX/Triton inference optimization pipeline")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp32", "fp16", "int8", "fp8"])
    parser.add_argument("--triton-repo", default="triton_repo",
                        help="Triton model repository output directory")
    parser.add_argument("--artifacts-dir", default="artifacts/trt",
                        help="Directory for ONNX and TRT artefacts")
    parser.add_argument("--triton-only", action="store_true",
                        help="Generate Triton config only (skip model loading)")
    args = parser.parse_args()

    cfg = OptimizationConfig(
        model_name=args.model,
        precision=args.precision,
        artifacts_dir=args.artifacts_dir,
    )

    if args.triton_only:
        gen = TritonConfigGenerator(cfg)
        gen.generate(args.triton_repo, backend="tensorrt")
        logger.info("Triton config generated at %s", args.triton_repo)
    else:
        pipeline = InferenceOptimizationPipeline(cfg)
        pipeline.run_full_pipeline(triton_repo=args.triton_repo)
