"""
Python wrapper for LLMOps custom CUDA kernels.

Provides:
  - Graceful fallback to pure-PyTorch implementations when CUDA extension
    is not compiled (CPU development, CI, etc.)
  - Type-checked entry points matching the CUDA kernel signatures
  - JIT compilation via torch.utils.cpp_extension.load() for fast iteration

Usage:
    from csrc.kernels_wrapper import (
        fused_attention_score_clip,
        top_k_sampling,
        rms_norm_fused,
        CUDA_KERNELS_AVAILABLE,
    )

    # These work on CPU too (fallback) — CUDA kernels activate automatically
    attn = fused_attention_score_clip(scores, clip_val=50.0, scale=0.125)
    tokens = top_k_sampling(logits, k=50, temperature=0.9)
    normed = rms_norm_fused(hidden_states, weight)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CUDA_KERNELS_AVAILABLE = False
_ext = None


def _try_load_extension():
    """Attempt to load the compiled CUDA extension."""
    global _ext, CUDA_KERNELS_AVAILABLE

    # 1. Try pre-compiled wheel
    try:
        import llmops_kernels as _ext
        CUDA_KERNELS_AVAILABLE = True
        logger.info("LLMOps CUDA kernels loaded (pre-compiled)")
        return
    except ImportError:
        pass

    # 2. Try JIT compilation (requires CUDA toolkit)
    if not torch.cuda.is_available():
        logger.info("CUDA not available — using PyTorch fallbacks for custom kernels")
        return

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = Path(__file__).parent
        _ext = load(
            name="llmops_kernels",
            sources=[str(csrc_dir / "llmops_kernels.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        CUDA_KERNELS_AVAILABLE = True
        logger.info("LLMOps CUDA kernels JIT-compiled successfully")
    except Exception as e:
        logger.warning("CUDA kernel JIT compilation failed — using PyTorch fallbacks: %s", e)


_try_load_extension()


# ──────────────────────────────────────────────────────────────────────────────
# Public API — CUDA kernel or PyTorch fallback
# ──────────────────────────────────────────────────────────────────────────────

def fused_attention_score_clip(
    scores: torch.Tensor,
    clip_val: float = 50.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Fused attention score scaling + clipping + softmax.

    Args:
        scores:   [B, H, Sq, Sk] float32 — raw QK^T attention scores
        clip_val: absolute clip threshold applied before softmax
        scale:    multiply scores by this before clipping (typically 1/sqrt(d_k))

    Returns:
        [B, H, Sq, Sk] float32 — softmax attention weights
    """
    if CUDA_KERNELS_AVAILABLE and scores.is_cuda:
        return _ext.fused_attention_score_clip(scores, clip_val, scale)

    # PyTorch fallback
    scaled = scores * scale
    clipped = scaled.clamp(-clip_val, clip_val)
    return F.softmax(clipped, dim=-1)


def top_k_sampling(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
    rand_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Top-k sampling from logit distributions.

    Args:
        logits:    [B, V] float32 — raw logits
        k:         number of top candidates to consider
        temperature: softmax temperature (lower = more greedy)
        rand_vals: [B] float32 uniform random values; generated if None

    Returns:
        [B] int32 — sampled token ids
    """
    B = logits.size(0)
    if rand_vals is None:
        rand_vals = torch.rand(B, device=logits.device, dtype=torch.float32)

    if CUDA_KERNELS_AVAILABLE and logits.is_cuda:
        return _ext.top_k_sampling(logits, k, temperature, rand_vals)

    # PyTorch fallback
    scaled = logits / temperature
    topk_vals, topk_idx = torch.topk(scaled, k, dim=-1)
    probs = F.softmax(topk_vals, dim=-1)
    # Multinomial sampling
    sampled_local = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return topk_idx.gather(-1, sampled_local.unsqueeze(-1)).squeeze(-1).to(torch.int32)


def rms_norm_fused(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm: output = (x / rms(x)) * weight

    Args:
        input:  [N, D] float32 — input hidden states
        weight: [D] float32 — learnable scale parameter
        eps:    small constant for numerical stability

    Returns:
        [N, D] float32 — normalised hidden states
    """
    if CUDA_KERNELS_AVAILABLE and input.is_cuda:
        return _ext.rms_norm_fused(input, weight, eps)

    # PyTorch fallback
    variance = input.pow(2).mean(-1, keepdim=True)
    normed = input * torch.rsqrt(variance + eps)
    return normed * weight


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark: CUDA kernel vs PyTorch baseline
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_kernels(device: str = "cuda", runs: int = 100) -> None:
    """Compare custom CUDA kernels against PyTorch baselines."""
    import time

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — running CPU benchmark only")
        device = "cpu"

    print(f"\nKernel benchmark — device={device}  CUDA_KERNELS={CUDA_KERNELS_AVAILABLE}")
    print("-" * 70)

    def timed(fn, *args, warmup=5):
        for _ in range(warmup):
            fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / runs * 1000

    # 1. fused_attention_score_clip
    B, H, Sq, Sk = 4, 8, 64, 512
    scores = torch.randn(B, H, Sq, Sk, device=device)

    def torch_attn_clip(s):
        return F.softmax(s.clamp(-50, 50), dim=-1)

    t_torch = timed(torch_attn_clip, scores)
    t_fused = timed(lambda s: fused_attention_score_clip(s, 50.0, 1.0), scores)
    print(f"fused_attention_score_clip  torch={t_torch:.3f}ms  custom={t_fused:.3f}ms  "
          f"speedup={t_torch/t_fused:.2f}x")

    # 2. top_k_sampling
    B, V = 16, 32000
    logits = torch.randn(B, V, device=device)
    rand_vals = torch.rand(B, device=device)

    def torch_topk_sample(l, rv):
        topk_v, topk_i = torch.topk(l, 50, dim=-1)
        probs = F.softmax(topk_v, dim=-1)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        return topk_i.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    t_torch = timed(torch_topk_sample, logits, rand_vals)
    t_fused = timed(lambda l, rv: top_k_sampling(l, k=50, rand_vals=rv), logits, rand_vals)
    print(f"top_k_sampling              torch={t_torch:.3f}ms  custom={t_fused:.3f}ms  "
          f"speedup={t_torch/t_fused:.2f}x")

    # 3. rms_norm_fused
    N, D = 512, 4096
    x = torch.randn(N, D, device=device)
    w = torch.ones(D, device=device)

    def torch_rms_norm(x, w):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * w

    t_torch = timed(torch_rms_norm, x, w)
    t_fused = timed(rms_norm_fused, x, w)
    print(f"rms_norm_fused              torch={t_torch:.3f}ms  custom={t_fused:.3f}ms  "
          f"speedup={t_torch/t_fused:.2f}x")

    print("-" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMOps CUDA kernel benchmarks")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    benchmark_kernels(device=args.device, runs=args.runs)
