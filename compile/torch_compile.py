"""
torch.compile and Ahead-of-Time (AoT) compilation pipeline.

Covers:
  - torch.compile() with multiple backends (inductor, cudagraphs, onnxrt)
  - AOT Autograd — trace and compile forward+backward graphs ahead of time
  - torch.export — full program capture for deployment
  - Dynamic shape handling — avoid recompilation on variable sequence lengths
  - Compilation profiling — warm-up, cache hits, graph breaks
  - Side-by-side benchmark: eager vs compiled (CPU + GPU)
  - Integration with existing vLLM and RAG inference pipeline

Typical speedups:
  Encoder models (BERT-style): 1.5–2.5x on GPU
  Decoder models (inference):  1.2–1.8x with cudagraphs
  Training backward pass:      1.3–2.0x with inductor

Usage:
    compiler = ModelCompiler(model)
    compiled = compiler.compile(backend="inductor", mode="reduce-overhead")
    results  = compiler.benchmark(inputs, n_runs=200)
    compiler.print_benchmark(results)

    # AoT export for deployment
    aot = AoTAutogradCompiler(model)
    ep  = aot.export(example_inputs, dynamic_shapes=DynamicShapeManager.build_dynamic_shapes())
    aot.save_exported("outputs/embedding.pt2")

    # Drop-in compilation for the RAG pipeline
    compiled_models = compile_rag_components(embedding_model, reranker_model)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompileConfig:
    backend: str = "inductor"       # inductor | cudagraphs | onnxrt | eager
    mode: str = "reduce-overhead"   # default | reduce-overhead | max-autotune
    dynamic: bool = True            # handle variable sequence lengths without recompile
    fullgraph: bool = False         # True = strict mode, fail on any graph break
    disable: bool = False           # True = skip compilation (A/B baseline)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompileBenchmarkResult:
    mode: str               # "eager" | "compiled"
    backend: str
    compile_mode: str
    warmup_runs: int
    bench_runs: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    speedup_vs_eager: float = 1.0
    graph_breaks: int = 0
    compile_time_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.mode:10s} [{self.backend}/{self.compile_mode}] | "
            f"p50={self.p50_ms:.1f}ms  p99={self.p99_ms:.1f}ms | "
            f"QPS={self.throughput_qps:.1f} | "
            f"speedup={self.speedup_vs_eager:.2f}x | "
            f"compile={self.compile_time_s:.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Graph break detector
# ──────────────────────────────────────────────────────────────────────────────

class GraphBreakDetector:
    """
    Detect and log torch.compile graph breaks.

    Graph breaks occur when the compiler encounters dynamic Python that it
    cannot trace (e.g. data-dependent control flow, unsupported ops).
    Each break forces a fallback to eager mode for that subgraph, reducing
    the achievable speedup.

    Use explain() before benchmarking to understand what's blocking full
    graph compilation — then fix the breaks or use fullgraph=False.
    """

    def __init__(self):
        self.breaks: List[str] = []

    def install_hook(self) -> None:
        """Enable verbose graph break logging via torch._dynamo."""
        try:
            import torch._dynamo as dynamo
            dynamo.config.verbose = True
            logger.info("Graph break detection enabled (torch._dynamo.config.verbose=True)")
        except Exception as e:
            logger.warning("Could not install graph break hook: %s", e)

    def explain(self, model, *args, **kwargs) -> Dict:
        """
        Use torch._dynamo.explain() to get a detailed compilation report.

        Returns:
            graphs:        number of separate compiled subgraphs
            graph_breaks:  number of breaks (ideally 0 for maximum speedup)
            break_reasons: human-readable reasons for each break
            ops_per_graph: operator count per subgraph
        """
        try:
            import torch

            explanation = torch._dynamo.explain(model)(*args, **kwargs)
            report = {
                "graphs": len(explanation.graphs),
                "graph_breaks": explanation.graph_break_count,
                "break_reasons": [str(r) for r in explanation.break_reasons],
                "ops_per_graph": [len(g.nodes) for g in explanation.graphs],
            }
            if report["graph_breaks"] > 0:
                logger.warning(
                    "%d graph break(s) detected — speedup will be limited.\n"
                    "Reasons: %s",
                    report["graph_breaks"],
                    report["break_reasons"],
                )
            else:
                logger.info("No graph breaks — full graph compilation achieved.")
            return report

        except Exception as e:
            logger.warning("explain() failed (requires PyTorch >= 2.1): %s", e)
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# Model compiler
# ──────────────────────────────────────────────────────────────────────────────

class ModelCompiler:
    """
    Wraps PyTorch models with torch.compile for inference and training speedup.

    Handles:
      - Multiple backends and modes
      - Warmup (first N runs trigger JIT kernel compilation)
      - Dynamic shape marking (avoids recompilation on variable inputs)
      - Benchmark comparison: eager vs all compiled variants
      - MLflow result logging
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self._compiled: Dict[str, Any] = {}
        self.detector = GraphBreakDetector()

    def compile(
        self,
        backend: str = "inductor",
        mode: str = "reduce-overhead",
        dynamic: bool = True,
        fullgraph: bool = False,
    ) -> Any:
        """
        Compile model with torch.compile.
        Caches compiled models by (backend, mode, dynamic) key.
        Returns compiled model, or original model if compilation fails.
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                logger.warning("torch.compile requires PyTorch >= 2.0 — using eager.")
                return self.model

            key = f"{backend}_{mode}_{dynamic}"
            if key in self._compiled:
                logger.debug("Returning cached compiled model: %s", key)
                return self._compiled[key]

            logger.info(
                "Compiling: backend=%s  mode=%s  dynamic=%s  fullgraph=%s",
                backend, mode, dynamic, fullgraph,
            )
            t0 = time.perf_counter()
            compiled = torch.compile(
                self.model,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )
            compile_time = time.perf_counter() - t0
            self._compiled[key] = compiled
            logger.info("torch.compile finished in %.2fs", compile_time)
            return compiled

        except Exception as e:
            logger.warning("torch.compile failed (%s) — falling back to eager.", e)
            return self.model

    def compile_for_inference(self, model=None) -> Any:
        """
        Recommended settings for inference-only compilation.
        reduce-overhead applies cudagraphs on GPU, minimising Python dispatch cost.
        """
        m = model or self.model
        try:
            import torch
            return torch.compile(m, backend="inductor", mode="reduce-overhead", dynamic=True)
        except Exception:
            return m

    def compile_for_training(self, model=None) -> Any:
        """
        Recommended settings for training compilation.
        max-autotune performs exhaustive kernel search — slow to compile,
        fastest at runtime. Use static shapes (dynamic=False) for best results.
        """
        m = model or self.model
        try:
            import torch
            return torch.compile(m, backend="inductor", mode="max-autotune", dynamic=False)
        except Exception:
            return m

    def warmup(self, compiled_model, inputs: Dict, n_warmup: int = 20) -> None:
        """
        Run warmup iterations to trigger JIT kernel compilation.

        The first N runs are slow because inductor compiles CUDA kernels on demand.
        Warmup ensures these costs are paid before the benchmark window.
        Typical warmup: 10–30 iterations depending on model size.
        """
        import torch

        logger.info("Warming up: %d iterations …", n_warmup)
        with torch.no_grad():
            for i in range(n_warmup):
                try:
                    compiled_model(**inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning("Warmup iteration %d failed: %s", i, e)
                    break
        logger.info("Warmup complete.")

    def _run_latencies(
        self, model, inputs: Dict, n_runs: int
    ) -> List[float]:
        import torch

        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
        return latencies

    def benchmark(
        self,
        inputs: Dict,
        backends: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        n_warmup: int = 20,
        n_runs: int = 200,
    ) -> List[CompileBenchmarkResult]:
        """
        Benchmark eager vs multiple compiled (backend, mode) combinations.

        Args:
            inputs:   dict of model inputs (e.g. from tokenizer)
            backends: list of torch.compile backends to test
            modes:    list of inductor modes to test
            n_warmup: warmup iterations (not included in timing)
            n_runs:   timed iterations per configuration

        Returns:
            List of CompileBenchmarkResult, first entry is always eager baseline.
        """
        backends = backends or ["inductor"]
        modes = modes or ["reduce-overhead"]
        results: List[CompileBenchmarkResult] = []

        # Eager baseline
        self.warmup(self.model, inputs, n_warmup)
        eager_latencies = self._run_latencies(self.model, inputs, n_runs)
        eager_p50 = float(np.percentile(eager_latencies, 50))

        results.append(CompileBenchmarkResult(
            mode="eager",
            backend="pytorch",
            compile_mode="none",
            warmup_runs=n_warmup,
            bench_runs=n_runs,
            p50_ms=eager_p50,
            p90_ms=float(np.percentile(eager_latencies, 90)),
            p99_ms=float(np.percentile(eager_latencies, 99)),
            mean_ms=float(np.mean(eager_latencies)),
            throughput_qps=float(1000 / np.mean(eager_latencies)),
            speedup_vs_eager=1.0,
        ))

        # Compiled variants
        for backend in backends:
            for mode in modes:
                t_compile = time.perf_counter()
                compiled = self.compile(backend=backend, mode=mode)
                compile_time = time.perf_counter() - t_compile

                self.warmup(compiled, inputs, n_warmup)
                latencies = self._run_latencies(compiled, inputs, n_runs)
                p50 = float(np.percentile(latencies, 50))

                results.append(CompileBenchmarkResult(
                    mode="compiled",
                    backend=backend,
                    compile_mode=mode,
                    warmup_runs=n_warmup,
                    bench_runs=n_runs,
                    p50_ms=p50,
                    p90_ms=float(np.percentile(latencies, 90)),
                    p99_ms=float(np.percentile(latencies, 99)),
                    mean_ms=float(np.mean(latencies)),
                    throughput_qps=float(1000 / np.mean(latencies)),
                    speedup_vs_eager=eager_p50 / p50 if p50 > 0 else 1.0,
                    compile_time_s=compile_time,
                ))
                logger.info(results[-1])

        return results

    def print_benchmark(self, results: List[CompileBenchmarkResult]) -> None:
        print("\n" + "=" * 100)
        print(f"{'Mode':10s} {'Backend':12s} {'CompileMode':18s} "
              f"{'p50':>9s} {'p90':>9s} {'p99':>9s} {'QPS':>8s} {'Speedup':>9s} {'Compile':>9s}")
        print("=" * 100)
        for r in results:
            print(
                f"{r.mode:10s} {r.backend:12s} {r.compile_mode:18s} "
                f"{r.p50_ms:8.1f}ms {r.p90_ms:8.1f}ms {r.p99_ms:8.1f}ms "
                f"{r.throughput_qps:8.1f} {r.speedup_vs_eager:8.2f}x "
                f"{r.compile_time_s:8.1f}s"
            )
        print("=" * 100)
        if len(results) > 1:
            best = max(results[1:], key=lambda r: r.speedup_vs_eager)
            print(f"\nBest: {best.speedup_vs_eager:.2f}x speedup "
                  f"({best.backend}/{best.compile_mode})")

    def log_to_mlflow(
        self,
        results: List[CompileBenchmarkResult],
        experiment: str = "torch-compile-benchmark",
    ) -> None:
        try:
            import mlflow

            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name="compile-benchmark"):
                for r in results:
                    prefix = f"{r.mode}.{r.backend}.{r.compile_mode}"
                    mlflow.log_metrics({
                        f"{prefix}.p50_ms": r.p50_ms,
                        f"{prefix}.p99_ms": r.p99_ms,
                        f"{prefix}.speedup": r.speedup_vs_eager,
                        f"{prefix}.qps": r.throughput_qps,
                        f"{prefix}.compile_s": r.compile_time_s,
                    })
            logger.info("Benchmark results logged to MLflow: %s", experiment)
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# AoT Autograd compiler
# ──────────────────────────────────────────────────────────────────────────────

class AoTAutogradCompiler:
    """
    Ahead-of-Time compilation using torch.export + AOT Autograd.

    AoT Autograd traces the forward AND backward graphs at compile time,
    enabling:
      - Operator fusion across forward + backward passes
      - Memory layout optimisations (contiguous vs strided)
      - Elimination of Python overhead in the training loop

    torch.export produces a portable, serialisable ExportedProgram
    for deployment — no Python runtime needed at inference time.
    """

    def __init__(self, model):
        self.model = model
        self._exported: Any = None

    def export(
        self,
        example_inputs: Tuple,
        dynamic_shapes: Optional[Dict] = None,
    ) -> Any:
        """
        Export model to a portable ExportedProgram.

        Args:
            example_inputs:  tuple of example tensors (same as model's forward args)
            dynamic_shapes:  {arg_name: {dim_index: Dim(...)}} for variable shapes

        Returns:
            torch.export.ExportedProgram or None on failure
        """
        try:
            import torch

            logger.info("Exporting model with torch.export …")
            t0 = time.perf_counter()

            self._exported = torch.export.export(
                self.model,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
            )

            export_time = time.perf_counter() - t0
            logger.info("torch.export complete in %.2fs", export_time)

            # Log graph statistics
            n_nodes = len(list(self._exported.graph.nodes))
            logger.info("Exported graph: %d nodes", n_nodes)

            return self._exported

        except Exception as e:
            logger.warning("torch.export failed: %s", e)
            return None

    def compile_exported(self, exported=None) -> Any:
        """
        Compile an ExportedProgram with inductor for maximum performance.
        max-autotune performs exhaustive kernel search on the static graph.
        """
        try:
            import torch

            ep = exported or self._exported
            if ep is None:
                raise ValueError("No exported program — call .export() first.")
            return torch.compile(ep.module(), backend="inductor", mode="max-autotune")
        except Exception as e:
            logger.warning("Compiled export failed: %s", e)
            return self.model

    def save_exported(self, path: str, exported=None) -> None:
        """Serialise ExportedProgram to disk for deployment."""
        try:
            import torch
            import os

            ep = exported or self._exported
            if ep is None:
                raise ValueError("No exported program to save.")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.export.save(ep, path)
            logger.info("ExportedProgram saved: %s", path)
        except Exception as e:
            logger.warning("Export save failed: %s", e)

    def load_exported(self, path: str) -> Any:
        """Load a previously saved ExportedProgram."""
        try:
            import torch

            self._exported = torch.export.load(path)
            logger.info("ExportedProgram loaded: %s", path)
            return self._exported
        except Exception as e:
            logger.warning("Export load failed: %s", e)
            return None

    def aot_eager_trace(self, example_inputs: Tuple) -> Any:
        """
        Trace forward + backward graphs using AOT Autograd with the 'eager'
        backend — no kernel compilation, just graph inspection.
        Useful for debugging fusions and understanding what inductor will see.
        """
        try:
            from torch._functorch.aot_autograd import aot_module_simplified
            import torch

            def fw_compiler(fx_module, inputs):
                n_ops = len([n for n in fx_module.graph.nodes if n.op == "call_function"])
                logger.info("AOT forward graph: %d ops", n_ops)
                return fx_module

            def bw_compiler(fx_module, inputs):
                n_ops = len([n for n in fx_module.graph.nodes if n.op == "call_function"])
                logger.info("AOT backward graph: %d ops", n_ops)
                return fx_module

            traced = aot_module_simplified(
                self.model,
                example_inputs,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
            )
            return traced
        except ImportError:
            logger.warning("torch._functorch not available — skipping AOT trace.")
            return self.model
        except Exception as e:
            logger.warning("AOT eager trace failed: %s", e)
            return self.model


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic shape manager
# ──────────────────────────────────────────────────────────────────────────────

class DynamicShapeManager:
    """
    Manage dynamic shapes for torch.compile to avoid recompilation when
    input dimensions change (variable batch size, sequence length).

    Without dynamic shapes, torch.compile recompiles for every new
    (batch_size, seq_len) pair — expensive in production where inputs vary.
    With dynamic shapes, a single compiled graph handles all sizes in the
    specified range.
    """

    @staticmethod
    def build_dynamic_shapes(
        batch_range: Tuple[int, int] = (1, 32),
        seq_range: Tuple[int, int] = (1, 2048),
    ) -> Dict:
        """
        Build dynamic shape spec for transformer models.

        Returns a dict compatible with torch.export(dynamic_shapes=...) and
        torch.compile(dynamic=True).
        """
        try:
            import torch

            batch_dim = torch.export.Dim("batch", min=batch_range[0], max=batch_range[1])
            seq_dim = torch.export.Dim("seq_len", min=seq_range[0], max=seq_range[1])
            return {
                "input_ids": {0: batch_dim, 1: seq_dim},
                "attention_mask": {0: batch_dim, 1: seq_dim},
            }
        except Exception as e:
            logger.warning("Could not build dynamic shapes: %s", e)
            return {}

    @staticmethod
    def mark_dynamic(tensor, dims: List[int]) -> None:
        """
        Mark specific tensor dimensions as dynamic for torch.compile.
        Call before the first compiled forward pass.
        """
        try:
            import torch

            for dim in dims:
                torch._dynamo.mark_dynamic(tensor, dim)
        except Exception:
            pass

    @staticmethod
    def reset_cache() -> None:
        """Clear the torch.compile compilation cache (force recompile)."""
        try:
            import torch
            torch._dynamo.reset()
            logger.info("torch._dynamo cache cleared.")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# RAG pipeline integration
# ──────────────────────────────────────────────────────────────────────────────

def compile_rag_components(
    embedding_model,
    reranker_model=None,
    reward_model=None,
    device: str = "cpu",
    backend: str = "inductor",
    mode: str = "reduce-overhead",
) -> Dict[str, Any]:
    """
    Apply torch.compile to RAG pipeline components.
    Returns a dict of compiled models for drop-in replacement.

    Usage:
        from compile.torch_compile import compile_rag_components
        compiled = compile_rag_components(embedder, reranker, device="cuda")
        # Replace originals:
        retriever.model = compiled["embedding"]
        reranker.model  = compiled["reranker"]
    """
    compiled: Dict[str, Any] = {}

    for name, model in [
        ("embedding", embedding_model),
        ("reranker", reranker_model),
        ("reward", reward_model),
    ]:
        if model is None:
            continue
        compiler = ModelCompiler(model, device=device)
        compiled[name] = compiler.compile(backend=backend, mode=mode)
        logger.info("%s model compiled (%s/%s)", name, backend, mode)

    return compiled


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="torch.compile benchmark suite")
    parser.add_argument("--model", default="prajjwal1/bert-tiny",
                        help="HuggingFace model name or local path")
    parser.add_argument("--backends", nargs="+", default=["inductor"],
                        choices=["inductor", "eager", "aot_eager", "cudagraphs"])
    parser.add_argument("--modes", nargs="+", default=["default", "reduce-overhead"],
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--export-aot", action="store_true",
                        help="Export model with torch.export after benchmarking")
    parser.add_argument("--graph-breaks", action="store_true",
                        help="Run graph break analysis before benchmarking")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log results to MLflow")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        args.device = "cpu"

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()

    dummy = tokenizer(
        "Explain retrieval-augmented generation in detail.",
        return_tensors="pt",
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
    )
    dummy = {k: v.to(args.device) for k, v in dummy.items()}

    compiler = ModelCompiler(model, device=args.device)

    # Graph break analysis
    if args.graph_breaks:
        detector = GraphBreakDetector()
        report = detector.explain(model, **dummy)
        if report:
            print(f"\nGraph break report:")
            print(f"  Graphs:       {report.get('graphs', '?')}")
            print(f"  Graph breaks: {report.get('graph_breaks', '?')}")
            for reason in report.get("break_reasons", []):
                print(f"  Break reason: {reason}")

    # Benchmark
    results = compiler.benchmark(
        dummy,
        backends=args.backends,
        modes=args.modes,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )
    compiler.print_benchmark(results)

    if args.mlflow:
        compiler.log_to_mlflow(results)

    # AoT export
    if args.export_aot:
        aot = AoTAutogradCompiler(model)
        dynamic_shapes = DynamicShapeManager.build_dynamic_shapes()
        example = (dummy["input_ids"], dummy["attention_mask"])
        ep = aot.export(example, dynamic_shapes=dynamic_shapes)
        if ep is not None:
            aot.save_exported("outputs/aot/model.pt2")
