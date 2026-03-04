"""
Quantization Suite — PTQ, QAT, GPTQ, AWQ, and Sparsity
=========================================================
Covers the full spectrum of model compression techniques:

  PostTrainingQuantizer   — Post-Training Quantization (PTQ) via bitsandbytes
                            and PyTorch native quantisation (dynamic + static)
  GPTQQuantizer           — GPTQ 4-bit weight-only quantisation (AutoGPTQ)
  AWQQuantizer            — Activation-aware Weight Quantization (AutoAWQ)
  QATTrainer              — Quantization-Aware Training via PyTorch FX graph mode
  SparsityPruner          — Unstructured and structured pruning + magnitude pruning
  QuantizationBenchmark   — Compares model size, latency, and quality across methods

Terminology:
  PTQ  — quantise after training; fast, slight quality loss
  QAT  — simulate quantisation during training; better quality, slower
  GPTQ — layer-wise second-order PTQ; near-lossless 4-bit for LLMs
  AWQ  — activation-aware scaling before quantisation; best 4-bit quality
  Sparsity — zero out small weights; reduces memory with sparse kernels

Run locally (CPU, no GPU needed for PTQ demo):
    python finetune/quantization_suite.py
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    model_name: str = "facebook/opt-125m"       # small model for demos
    output_dir: str = "outputs/quantized"
    calibration_samples: int = 128
    calibration_seq_len: int = 512
    # PTQ
    ptq_bits: int = 8                           # 4 or 8
    # GPTQ
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    # AWQ
    awq_bits: int = 4
    awq_group_size: int = 128
    # QAT
    qat_epochs: int = 1
    qat_lr: float = 1e-5
    qat_batch_size: int = 4
    # Sparsity
    sparsity_ratio: float = 0.5                 # fraction of weights to zero
    structured_sparsity: bool = False           # True = channel pruning
    # Benchmark
    benchmark_prompt: str = "Explain retrieval-augmented generation in detail."
    benchmark_max_tokens: int = 128
    benchmark_n_runs: int = 10


# ---------------------------------------------------------------------------
# Post-Training Quantization (PTQ)
# ---------------------------------------------------------------------------

class PostTrainingQuantizer:
    """
    Post-Training Quantization using:
      1. bitsandbytes NF4/INT8 (HuggingFace integration)
      2. PyTorch dynamic quantisation (CPU, no calibration needed)
      3. PyTorch static quantisation (requires calibration dataset)

    PTQ is the fastest path — no retraining required.
    Quality loss is small for INT8; noticeable for INT4 without GPTQ/AWQ.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize_bnb_int8(self, model_name: Optional[str] = None) -> Any:
        """
        Load model in INT8 using bitsandbytes LLM.int8().
        Reduces memory by ~2x vs FP16 with minimal quality loss.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = model_name or self.config.model_name
        logger.info("Loading INT8 quantised model: %s", model_name)

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("INT8 model loaded. Memory: %.2f GB", self._model_memory_gb(model))
        return model, tokenizer

    def quantize_bnb_nf4(
        self,
        model_name: Optional[str] = None,
        double_quant: bool = True,
    ) -> Any:
        """
        Load model in NF4 (4-bit Normal Float) using bitsandbytes.
        This is the same quantisation used by QLoRA fine-tuning.
        double_quant=True applies a second quantisation to the scale factors,
        saving an additional ~0.4 bits per parameter.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = model_name or self.config.model_name
        logger.info("Loading NF4 quantised model: %s (double_quant=%s)", model_name, double_quant)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("NF4 model loaded. Memory: %.2f GB", self._model_memory_gb(model))
        return model, tokenizer

    def quantize_pytorch_dynamic(self, model: nn.Module) -> nn.Module:
        """
        PyTorch dynamic quantisation — quantises weights to INT8 at load time,
        activations quantised dynamically at runtime.
        No calibration data required. Works on CPU only.
        Best for LSTM/Linear-heavy models.
        """
        logger.info("Applying PyTorch dynamic quantisation (INT8, CPU)")
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )
        return quantized

    def quantize_pytorch_static(
        self,
        model: nn.Module,
        calibration_loader,
        backend: str = "fbgemm",  # "fbgemm" (x86) or "qnnpack" (ARM)
    ) -> nn.Module:
        """
        PyTorch static quantisation — calibrates scale/zero-point from data.
        Better accuracy than dynamic quantisation for CNN/transformer models.
        Requires a calibration dataset.
        """
        logger.info("Applying PyTorch static quantisation (backend=%s)", backend)
        torch.backends.quantized.engine = backend

        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)

        # Calibration pass
        logger.info("Running calibration (%d batches)...", self.config.calibration_samples)
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= self.config.calibration_samples:
                    break
                model(**batch)

        torch.quantization.convert(model, inplace=True)
        logger.info("Static quantisation complete")
        return model

    @staticmethod
    def _model_memory_gb(model: nn.Module) -> float:
        try:
            params = sum(p.numel() * p.element_size() for p in model.parameters())
            buffers = sum(b.numel() * b.element_size() for b in model.buffers())
            return (params + buffers) / (1024 ** 3)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# GPTQ Quantization
# ---------------------------------------------------------------------------

class GPTQQuantizer:
    """
    GPTQ (Generative Pre-Trained Transformer Quantization).

    Layer-wise second-order PTQ that minimises per-layer reconstruction error.
    Achieves near-lossless 4-bit quantisation for large LLMs.
    Requires: auto-gptq (pip install auto-gptq)

    Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization
    for Generative Pre-trained Transformers" (2022).
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(
        self,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        calibration_texts: Optional[List[str]] = None,
    ) -> str:
        """
        Quantise a model to GPTQ 4-bit and save to disk.

        Args:
            model_name:         HuggingFace model ID.
            output_dir:         Where to save the quantised model.
            calibration_texts:  Representative text samples for calibration.

        Returns:
            Path to the saved quantised model.
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError("Install auto-gptq: pip install auto-gptq")

        from transformers import AutoTokenizer
        from datasets import Dataset

        model_name = model_name or self.config.model_name
        output_dir = output_dir or os.path.join(self.config.output_dir, "gptq")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("GPTQ quantisation: %s → %d-bit", model_name, self.config.gptq_bits)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Calibration dataset
        if calibration_texts is None:
            calibration_texts = self._default_calibration_texts()

        examples = [
            tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.calibration_seq_len,
                truncation=True,
            )
            for text in calibration_texts[: self.config.calibration_samples]
        ]

        quant_config = BaseQuantizeConfig(
            bits=self.config.gptq_bits,
            group_size=self.config.gptq_group_size,
            desc_act=self.config.gptq_desc_act,
        )

        model = AutoGPTQForCausalLM.from_pretrained(model_name, quant_config)
        model.quantize(examples)
        model.save_quantized(output_dir, use_safetensors=True)
        tokenizer.save_pretrained(output_dir)

        logger.info("GPTQ model saved: %s", output_dir)
        return output_dir

    def load(self, model_dir: str) -> Any:
        """Load a previously quantised GPTQ model."""
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        logger.info("Loading GPTQ model: %s", model_dir)
        model = AutoGPTQForCausalLM.from_quantized(
            model_dir,
            use_safetensors=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer

    @staticmethod
    def _default_calibration_texts() -> List[str]:
        return [
            "Retrieval-augmented generation combines a retrieval system with a generative model.",
            "Large language models are trained on vast corpora of text data.",
            "Quantization reduces model size by representing weights with fewer bits.",
            "The attention mechanism allows transformers to focus on relevant context.",
            "Fine-tuning adapts a pre-trained model to a specific downstream task.",
        ] * 30


# ---------------------------------------------------------------------------
# AWQ Quantization
# ---------------------------------------------------------------------------

class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization).

    Identifies salient weights using activation statistics and applies
    per-channel scaling before quantisation. Achieves better quality than
    GPTQ at 4-bit for instruction-following models.
    Requires: autoawq (pip install autoawq)

    Reference: Lin et al., "AWQ: Activation-aware Weight Quantization
    for LLM Compression and Acceleration" (2023).
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(
        self,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Quantise a model to AWQ 4-bit and save to disk.

        Returns:
            Path to the saved quantised model.
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("Install autoawq: pip install autoawq")

        from transformers import AutoTokenizer

        model_name = model_name or self.config.model_name
        output_dir = output_dir or os.path.join(self.config.output_dir, "awq")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("AWQ quantisation: %s → %d-bit", model_name, self.config.awq_bits)

        model = AutoAWQForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        quant_config = {
            "zero_point": True,
            "q_group_size": self.config.awq_group_size,
            "w_bit": self.config.awq_bits,
            "version": "GEMM",
        }

        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("AWQ model saved: %s", output_dir)
        return output_dir

    def load(self, model_dir: str) -> Any:
        """Load a previously quantised AWQ model."""
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        logger.info("Loading AWQ model: %s", model_dir)
        model = AutoAWQForCausalLM.from_quantized(model_dir, fuse_layers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer


# ---------------------------------------------------------------------------
# Quantization-Aware Training (QAT)
# ---------------------------------------------------------------------------

class QATTrainer:
    """
    Quantization-Aware Training (QAT) via PyTorch FX graph mode.

    Inserts fake-quantisation nodes during forward pass so the model
    learns to be robust to quantisation error. Produces better quality
    than PTQ, especially at INT4, at the cost of additional training time.

    Best for: small models, edge deployment, when PTQ quality is insufficient.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def prepare(self, model: nn.Module, backend: str = "fbgemm") -> nn.Module:
        """
        Insert fake-quantisation nodes into the model graph.
        Call this before training, then call convert() after.
        """
        import torch.ao.quantization as tq

        logger.info("Preparing model for QAT (backend=%s)", backend)
        model.train()
        model.qconfig = tq.get_default_qat_qconfig(backend)
        tq.prepare_qat(model, inplace=True)
        return model

    def train(
        self,
        model: nn.Module,
        train_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> nn.Module:
        """
        Run QAT training loop with fake-quantisation active.

        Args:
            model:        Model prepared with prepare().
            train_loader: DataLoader yielding (input_ids, labels) batches.
            optimizer:    Optional custom optimizer.

        Returns:
            Trained model (still in fake-quant mode; call convert() next).
        """
        model.train()
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.qat_lr)

        logger.info("QAT training: %d epochs", self.config.qat_epochs)
        for epoch in range(self.config.qat_epochs):
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            logger.info("QAT epoch %d/%d — loss=%.4f", epoch + 1, self.config.qat_epochs, total_loss / max(steps, 1))

        return model

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Convert fake-quantisation nodes to real INT8 operations.
        Call after QAT training is complete.
        """
        import torch.ao.quantization as tq

        model.eval()
        tq.convert(model, inplace=True)
        logger.info("QAT conversion complete — model is now INT8")
        return model


# ---------------------------------------------------------------------------
# Sparsity / Pruning
# ---------------------------------------------------------------------------

class SparsityPruner:
    """
    Weight pruning for model compression.

    Techniques:
      - Unstructured (magnitude) pruning: zeroes individual weights below threshold
      - Structured (channel) pruning: removes entire neurons/channels
      - Gradual magnitude pruning: increases sparsity over training steps

    Sparse models require sparse kernels (e.g. NVIDIA 2:4 sparsity, DeepSparse)
    to realise inference speedups. Without sparse kernels, pruning reduces
    memory but not necessarily latency.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def prune_unstructured(
        self,
        model: nn.Module,
        sparsity: Optional[float] = None,
        layers: Optional[List[type]] = None,
    ) -> nn.Module:
        """
        Global magnitude pruning — zeroes the smallest weights globally.
        Achieves high sparsity with minimal quality loss up to ~50%.
        """
        import torch.nn.utils.prune as prune

        sparsity = sparsity or self.config.sparsity_ratio
        layers = layers or [nn.Linear]

        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, tuple(layers))
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Make pruning permanent (remove masks)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        actual_sparsity = self._compute_sparsity(model)
        logger.info(
            "Unstructured pruning complete. Target=%.1f%%, Actual=%.1f%%",
            sparsity * 100, actual_sparsity * 100,
        )
        return model

    def prune_structured(
        self,
        model: nn.Module,
        sparsity: Optional[float] = None,
    ) -> nn.Module:
        """
        Structured (channel/neuron) pruning — removes entire output channels.
        Produces dense models that benefit from standard BLAS kernels.
        """
        import torch.nn.utils.prune as prune

        sparsity = sparsity or self.config.sparsity_ratio

        for module in model.modules():
            if isinstance(module, nn.Linear) and module.out_features > 1:
                prune.ln_structured(
                    module, name="weight", amount=sparsity, n=2, dim=0
                )
                prune.remove(module, "weight")

        actual_sparsity = self._compute_sparsity(model)
        logger.info(
            "Structured pruning complete. Target=%.1f%%, Actual=%.1f%%",
            sparsity * 100, actual_sparsity * 100,
        )
        return model

    def apply_nvidia_2_4_sparsity(self, model: nn.Module) -> nn.Module:
        """
        Apply NVIDIA 2:4 structured sparsity pattern.
        Exactly 2 out of every 4 consecutive weights are zeroed.
        NVIDIA Ampere+ GPUs have hardware support for 2:4 sparse matrix multiply,
        giving 2x speedup with near-lossless quality.
        Requires: torch >= 2.1 with CUDA 11.8+
        """
        try:
            from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
            logger.info("Applying NVIDIA 2:4 semi-structured sparsity")
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight = nn.Parameter(
                        to_sparse_semi_structured(module.weight)
                    )
            logger.info("2:4 sparsity applied")
        except (ImportError, AttributeError) as e:
            logger.warning("2:4 sparsity not available (requires PyTorch 2.1+ with CUDA): %s", e)
        return model

    @staticmethod
    def _compute_sparsity(model: nn.Module) -> float:
        total = 0
        zeros = 0
        for param in model.parameters():
            total += param.numel()
            zeros += (param == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    @staticmethod
    def sparsity_report(model: nn.Module) -> Dict[str, float]:
        """Per-layer sparsity report."""
        report = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight
                sparsity = (w == 0).float().mean().item()
                report[name] = round(sparsity, 4)
        return report


# ---------------------------------------------------------------------------
# Quantization Benchmark
# ---------------------------------------------------------------------------

class QuantizationBenchmark:
    """
    Compares model size, inference latency, and generation quality
    across quantisation methods. Logs results to MLflow.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def benchmark_model(
        self,
        model,
        tokenizer,
        label: str,
        mlflow_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure latency, throughput, and model size for a single model variant.

        Args:
            model:      The quantised (or baseline) model.
            tokenizer:  Corresponding tokenizer.
            label:      Name for this variant (e.g. "fp16", "int8", "gptq_4bit").

        Returns:
            Dict of benchmark metrics.
        """
        import mlflow

        device = next(model.parameters()).device
        inputs = tokenizer(
            self.config.benchmark_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model.generate(**inputs, max_new_tokens=32)

        # Latency measurement
        times = []
        tokens_generated = []
        with torch.no_grad():
            for _ in range(self.config.benchmark_n_runs):
                t0 = time.perf_counter()
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.config.benchmark_max_tokens,
                    do_sample=False,
                )
                elapsed = time.perf_counter() - t0
                times.append(elapsed * 1000)
                tokens_generated.append(out.shape[1] - inputs["input_ids"].shape[1])

        import numpy as np
        times_arr = np.array(times)
        tok_arr = np.array(tokens_generated)

        metrics = {
            "label": label,
            "p50_latency_ms": float(np.percentile(times_arr, 50)),
            "p99_latency_ms": float(np.percentile(times_arr, 99)),
            "mean_tok_per_sec": float((tok_arr / (times_arr / 1000)).mean()),
            "model_size_gb": PostTrainingQuantizer._model_memory_gb(model),
            "n_runs": self.config.benchmark_n_runs,
        }

        logger.info("Benchmark [%s]: p50=%.0fms, %.1f tok/s, %.2f GB",
                    label, metrics["p50_latency_ms"],
                    metrics["mean_tok_per_sec"], metrics["model_size_gb"])

        if mlflow_run:
            try:
                with mlflow.start_run(run_name=f"quant-{label}"):
                    mlflow.log_params({"label": label, "model": self.config.model_name})
                    mlflow.log_metrics({k: v for k, v in metrics.items()
                                        if isinstance(v, (int, float))})
            except Exception as e:
                logger.warning("MLflow logging failed: %s", e)

        return metrics

    def compare(self, variants: List[Tuple[Any, Any, str]]) -> List[Dict]:
        """
        Run benchmarks across multiple model variants and print a comparison table.

        Args:
            variants: List of (model, tokenizer, label) tuples.

        Returns:
            List of metric dicts, one per variant.
        """
        results = []
        for model, tokenizer, label in variants:
            metrics = self.benchmark_model(model, tokenizer, label)
            results.append(metrics)

        # Print comparison table
        print("\n" + "=" * 70)
        print(f"{'Method':<20} {'p50 (ms)':<12} {'tok/s':<10} {'Size (GB)':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['label']:<20} {r['p50_latency_ms']:<12.0f} "
                  f"{r['mean_tok_per_sec']:<10.1f} {r['model_size_gb']:<12.2f}")
        print("=" * 70)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Quantization Suite — PTQ / GPTQ / AWQ / QAT / Sparsity")
    logger.info("")

    config = QuantizationConfig(model_name="facebook/opt-125m")

    # Demonstrate PTQ dynamic quantisation (CPU, no download needed beyond opt-125m)
    logger.info("Demo: PyTorch dynamic quantisation on a small Linear model")
    small_model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )
    ptq = PostTrainingQuantizer(config)
    quantized = ptq.quantize_pytorch_dynamic(small_model)
    logger.info("Dynamic quantisation complete: %s", quantized)

    # Demonstrate sparsity
    logger.info("")
    logger.info("Demo: Unstructured magnitude pruning (50%% sparsity)")
    pruner = SparsityPruner(config)
    pruned = pruner.prune_unstructured(small_model, sparsity=0.5)
    report = SparsityPruner.sparsity_report(pruned)
    logger.info("Sparsity per layer: %s", report)

    logger.info("")
    logger.info("To run full quantisation pipelines:")
    logger.info("  GPTQ: GPTQQuantizer(config).quantize()  # requires auto-gptq + GPU")
    logger.info("  AWQ:  AWQQuantizer(config).quantize()   # requires autoawq + GPU")
    logger.info("  NF4:  PostTrainingQuantizer(config).quantize_bnb_nf4()  # requires bitsandbytes + GPU")
