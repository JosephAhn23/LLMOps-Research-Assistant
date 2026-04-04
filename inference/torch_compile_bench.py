"""
torch.compile / Ahead-of-Time (AoT) Compilation Benchmarking
=============================================================
Wraps existing PyTorch models with torch.compile() and measures
eager vs compiled speedup across multiple backends and modes.

Backends:
  - inductor   (default, best for CPU/GPU, fuses ops into Triton/C++ kernels)
  - eager      (baseline, no compilation)
  - aot_eager  (AoT Autograd only, no kernel fusion — good for debugging)
  - cudagraphs (CUDA graph capture, lowest GPU launch overhead)
  - onnxrt     (export to ONNX and run via ONNXRuntime)

Modes (inductor):
  - default          — balanced compile time vs runtime speedup
  - reduce-overhead  — minimise Python overhead (best for small batches)
  - max-autotune     — exhaustive kernel search (slowest compile, fastest runtime)

Usage:
    # CPU benchmark (no GPU required)
    python inference/torch_compile_bench.py --device cpu --model embedding

    # Full benchmark suite
    python inference/torch_compile_bench.py --device cuda --model all --runs 100

    # Compile a specific model and save
    python inference/torch_compile_bench.py --compile-only --model reranker
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompileBenchResult:
    model_name: str
    backend: str
    mode: str
    device: str
    batch_size: int
    seq_len: int
    eager_ms: float
    compiled_ms: float
    compile_time_s: float
    speedup: float = field(init=False)
    first_run_ms: float = 0.0   # includes compilation warm-up

    def __post_init__(self):
        self.speedup = self.eager_ms / self.compiled_ms if self.compiled_ms > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.model_name:<20} backend={self.backend:<12} mode={self.mode:<18} "
            f"device={self.device:<6} bs={self.batch_size} seq={self.seq_len} | "
            f"eager={self.eager_ms:.2f}ms  compiled={self.compiled_ms:.2f}ms  "
            f"speedup={self.speedup:.2f}x  compile={self.compile_time_s:.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model factories (lightweight stand-ins for the real pipeline models)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModel(nn.Module):
    """
    Lightweight sentence-embedding model (mirrors all-MiniLM-L6-v2 architecture).
    Used in the FAISS retrieval pipeline.
    """
    def __init__(self, vocab_size: int = 30522, hidden: int = 384, layers: int = 6, heads: int = 12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool = nn.Linear(hidden, hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(1)
        return torch.nn.functional.normalize(self.pool(x), dim=-1)


class CrossEncoderModel(nn.Module):
    """
    Cross-encoder reranker (mirrors ms-marco-MiniLM architecture).
    Scores query-document pairs for the reranking stage.
    """
    def __init__(self, vocab_size: int = 30522, hidden: int = 384, layers: int = 6, heads: int = 12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)


class RewardModel(nn.Module):
    """
    Reward model for RLHF scoring (mirrors deberta-v3-large reward model).
    Scores response quality for PPO training.
    """
    def __init__(self, vocab_size: int = 50265, hidden: int = 512, layers: int = 4, heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        x = self.encoder(x)
        return self.head(x[:, 0, :]).squeeze(-1)


MODEL_REGISTRY: Dict[str, Callable] = {
    "embedding": EmbeddingModel,
    "reranker": CrossEncoderModel,
    "reward": RewardModel,
}


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarking engine
# ──────────────────────────────────────────────────────────────────────────────

def _timed_run(
    model: nn.Module,
    inputs: Tuple,
    runs: int = 50,
    warmup: int = 5,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Returns (first_run_ms, mean_ms_after_warmup).
    First run captures compilation + execution time.
    """
    model.eval()
    with torch.no_grad():
        # First run (may include compilation)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        first_run_ms = (time.perf_counter() - t0) * 1000

        # Warmup
        for _ in range(warmup):
            model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(runs):
            model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        mean_ms = (time.perf_counter() - t0) / runs * 1000

    return first_run_ms, mean_ms


def benchmark_model(
    model_name: str,
    model_class: Callable,
    device: str = "cpu",
    batch_size: int = 8,
    seq_len: int = 128,
    runs: int = 50,
    backends: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
) -> List[CompileBenchResult]:
    """
    Benchmark eager vs torch.compile() for a single model class.
    Returns a list of CompileBenchResult, one per (backend, mode) combination.
    """
    if backends is None:
        backends = ["inductor"]
    if modes is None:
        modes = ["default", "reduce-overhead"]

    results: List[CompileBenchResult] = []

    # Build inputs
    model = model_class().to(device)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    inputs = (input_ids, attention_mask)

    # Eager baseline
    logger.info("Benchmarking eager baseline: %s", model_name)
    _, eager_ms = _timed_run(model, inputs, runs=runs, device=device)
    logger.info("  Eager: %.2f ms/batch", eager_ms)

    # Compiled variants
    for backend in backends:
        for mode in modes:
            if backend != "inductor" and mode != "default":
                continue  # mode only applies to inductor

            logger.info("Compiling %s backend=%s mode=%s ...", model_name, backend, mode)
            model_fresh = model_class().to(device)
            model_fresh.load_state_dict(model.state_dict())

            compile_kwargs: Dict = {"backend": backend, "fullgraph": False}
            if backend == "inductor":
                compile_kwargs["mode"] = mode

            t_compile_start = time.perf_counter()
            try:
                compiled = torch.compile(model_fresh, **compile_kwargs)
            except Exception as e:
                logger.warning("torch.compile failed (%s/%s): %s", backend, mode, e)
                continue
            compile_time_s = time.perf_counter() - t_compile_start

            first_ms, compiled_ms = _timed_run(compiled, inputs, runs=runs, device=device)

            result = CompileBenchResult(
                model_name=model_name,
                backend=backend,
                mode=mode,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                eager_ms=eager_ms,
                compiled_ms=compiled_ms,
                compile_time_s=compile_time_s,
                first_run_ms=first_ms,
            )
            results.append(result)
            logger.info("  %s", result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# AoT export helpers
# ──────────────────────────────────────────────────────────────────────────────

def export_aot(
    model: nn.Module,
    example_inputs: Tuple,
    output_path: str,
    dynamic_shapes: bool = True,
) -> str:
    """
    Export a model using torch.export (AoT) for deployment.
    Produces a portable .pt2 artifact that can be loaded without the model class.

    Requires PyTorch >= 2.1.
    """
    import os

    logger.info("Exporting model with torch.export (AoT) ...")
    model.eval()

    if dynamic_shapes:
        # Mark batch dimension as dynamic so the export handles variable batch sizes
        batch_dim = torch.export.Dim("batch", min=1, max=256)
        seq_dim = torch.export.Dim("seq", min=1, max=512)
        dynamic_shapes_spec = {
            "input_ids": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: seq_dim},
        }
    else:
        dynamic_shapes_spec = None

    try:
        exported = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes_spec,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.export.save(exported, output_path)
        logger.info("AoT export saved: %s", output_path)
        return output_path
    except Exception as e:
        logger.error("torch.export failed: %s", e)
        raise


def load_aot(path: str) -> torch.export.ExportedProgram:
    """Load a previously exported AoT model."""
    logger.info("Loading AoT export: %s", path)
    return torch.export.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# MLflow logging
# ──────────────────────────────────────────────────────────────────────────────

def log_results_to_mlflow(results: List[CompileBenchResult], experiment: str = "torch-compile-bench"):
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="compile-benchmark"):
            for r in results:
                prefix = f"{r.model_name}.{r.backend}.{r.mode}"
                mlflow.log_metrics({
                    f"{prefix}.eager_ms": r.eager_ms,
                    f"{prefix}.compiled_ms": r.compiled_ms,
                    f"{prefix}.speedup": r.speedup,
                    f"{prefix}.compile_time_s": r.compile_time_s,
                })
        logger.info("Results logged to MLflow experiment: %s", experiment)
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="torch.compile benchmarking suite")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="embedding",
        help="Model to benchmark",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["inductor"],
        choices=["inductor", "eager", "aot_eager", "cudagraphs"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["default", "reduce-overhead"],
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--export-aot", action="store_true", help="Export best model with torch.export")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument("--compile-only", action="store_true", help="Compile without benchmarking")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        args.device = "cpu"

    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    all_results: List[CompileBenchResult] = []

    for name in model_names:
        results = benchmark_model(
            model_name=name,
            model_class=MODEL_REGISTRY[name],
            device=args.device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runs=args.runs,
            backends=args.backends,
            modes=args.modes,
        )
        all_results.extend(results)

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Model':<20} {'Backend':<12} {'Mode':<18} {'Device':<6} {'BS':>4} {'Seq':>4} "
          f"{'Eager ms':>10} {'Compiled ms':>12} {'Speedup':>9} {'Compile s':>10}")
    print("-" * 120)
    for r in all_results:
        print(f"{r.model_name:<20} {r.backend:<12} {r.mode:<18} {r.device:<6} "
              f"{r.batch_size:>4} {r.seq_len:>4} "
              f"{r.eager_ms:>10.2f} {r.compiled_ms:>12.2f} "
              f"{r.speedup:>8.2f}x {r.compile_time_s:>10.1f}s")
    print("=" * 120)

    if all_results:
        best = max(all_results, key=lambda r: r.speedup)
        print(f"\nBest speedup: {best.speedup:.2f}x  ({best.model_name} / {best.backend} / {best.mode})")

    if args.export_aot and all_results:
        model_name = model_names[0]
        model = MODEL_REGISTRY[model_name]().to(args.device)
        input_ids = torch.randint(0, 1000, (args.batch_size, args.seq_len), device=args.device)
        mask = torch.ones_like(input_ids)
        export_aot(model, (input_ids, mask), f"outputs/aot/{model_name}.pt2")

    if args.mlflow:
        log_results_to_mlflow(all_results)


if __name__ == "__main__":
    main()
