"""
Mixture-of-Experts (MoE) Serving & Multi-GPU Inference
=======================================================
Production serving configuration for MoE models (Mixtral-8x7B, DeepSeek-MoE,
Qwen-MoE) with expert parallelism, tensor parallelism, and load balancing.

MoE key concept:
  Instead of one dense FFN per layer, MoE has N "expert" FFNs.
  A learned router sends each token to the top-K experts (usually K=2).
  Result: same compute as a dense model of 1/N the size, but N× more parameters.
  Mixtral-8x7B: 8 experts per layer, top-2 routing → 12.9B active params, 46.7B total.

Components:
  1. MoEArchConfig        — structured architecture metadata + named presets
  2. MoERouter            — token routing logic (top-k expert selection)
  3. MoELayer             — full MoE FFN layer (router + expert pool)
  4. ExpertParallelConfig — expert + tensor parallelism layout
  5. MoELoadBalancer      — monitors expert utilisation, rebalances routing
  6. MoEServingConfig     — high-level vLLM serving config with memory estimation
  7. DeepSpeedMoEConfig   — DeepSpeed-Inference config for MoE models
  8. RayServeMoEDeployment — Ray Serve multi-replica deployment
  9. MultiGPUInferenceConfig — transformers device_map + pipeline parallelism

Hardware note:
  Actual multi-GPU execution requires the corresponding hardware.
  All config classes are fully writable and match what production deployments use.
  The router and load balancer run on CPU for testing.

Usage:
    # High-level serving config with memory estimate
    cfg = MoEServingConfig.for_mixtral(n_gpus=4)
    print(cfg.generate_vllm_launch_cmd())
    print(cfg.estimate_gpu_memory_gb())

    # Configure expert parallelism for Mixtral-8x7B on 8 GPUs
    ep_cfg = ExpertParallelConfig(model="mixtral-8x7b", n_experts=8,
                                  n_gpus=8, ep_size=8, tp_size=1)
    ep_cfg.print_layout()

    # Monitor expert load
    balancer = MoELoadBalancer(n_experts=8)
    balancer.update(routing_weights)
    balancer.report()
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# MoE Router
# ──────────────────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """
    Top-k sparse MoE router (matches Mixtral/DeepSeek routing logic).

    Routes each token to the top-k experts based on learned gate weights.
    Implements:
      - Softmax routing (standard)
      - Sigmoid routing (DeepSeek-MoE style — independent expert probabilities)
      - Auxiliary load-balancing loss (prevents expert collapse)

    Args:
        hidden_size:  model hidden dimension
        n_experts:    total number of experts
        top_k:        experts activated per token (Mixtral: 2, DeepSeek: 6)
        routing_type: "softmax" | "sigmoid"
        aux_loss_coef: coefficient for load-balancing auxiliary loss
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        n_experts: int = 8,
        top_k: int = 2,
        routing_type: str = "softmax",
        aux_loss_coef: float = 0.01,
        capacity_factor: float = 1.25,   # expert capacity = capacity_factor * (tokens / n_experts)
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.routing_type = routing_type
        self.aux_loss_coef = aux_loss_coef
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(hidden_size, n_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch * seq_len, hidden_size]

        Returns:
            routing_weights:   [batch * seq_len, top_k] — expert weights (sum to 1)
            selected_experts:  [batch * seq_len, top_k] — expert indices
            router_logits:     [batch * seq_len, n_experts] — raw gate logits
            aux_loss:          scalar auxiliary load-balancing loss (or None)
        """
        router_logits = self.gate(hidden_states)   # [T, E]

        if self.routing_type == "sigmoid":
            # DeepSeek-MoE: independent expert probabilities, no competition
            routing_probs = torch.sigmoid(router_logits)
            routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        else:
            # Mixtral-style: softmax over all experts, then top-k
            routing_weights_full = F.softmax(router_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(
                routing_weights_full, self.top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Auxiliary load-balancing loss (Switch Transformer / ST-MoE)
        aux_loss = None
        if self.training and self.aux_loss_coef > 0:
            aux_loss = self._load_balance_loss(router_logits, selected_experts)

        return routing_weights, selected_experts, router_logits, aux_loss

    def _load_balance_loss(
        self, router_logits: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Switch Transformer auxiliary loss: encourages uniform expert utilisation.
        L_aux = alpha * sum_i(f_i * P_i) * n_experts
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i
        """
        T = router_logits.size(0)
        # Expert fraction (hard assignment)
        expert_mask = torch.zeros(T, self.n_experts, device=router_logits.device)
        expert_mask.scatter_(1, selected_experts[:, :1], 1.0)  # use top-1 for loss
        f = expert_mask.mean(0)  # [E]
        # Mean routing probability
        P = F.softmax(router_logits, dim=-1).mean(0)  # [E]
        return self.aux_loss_coef * self.n_experts * (f * P).sum()

    def get_routing_stats(self, router_logits: torch.Tensor) -> Dict:
        """Return expert utilisation statistics for monitoring."""
        with torch.no_grad():
            probs = F.softmax(router_logits, dim=-1)
            _, top_experts = torch.topk(probs, self.top_k, dim=-1)
            counts = torch.zeros(self.n_experts, device=router_logits.device)
            counts.scatter_add_(0, top_experts.flatten(),
                                torch.ones_like(top_experts.flatten(), dtype=torch.float))
            total = top_experts.numel()
            utilisation = (counts / total * 100).tolist()
        return {
            f"expert_{i}_pct": f"{u:.1f}%" for i, u in enumerate(utilisation)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Expert parallelism configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpertParallelConfig:
    """
    Expert parallelism (EP) + tensor parallelism (TP) layout for MoE models.

    EP: each GPU holds a subset of experts. Tokens are dispatched via
        all-to-all communication to the GPU holding the selected expert.
    TP: each expert's FFN is sharded across multiple GPUs.

    Mixtral-8x7B on 8 GPUs:
      ep_size=8, tp_size=1  → each GPU holds 1 expert, no TP
      ep_size=4, tp_size=2  → 4 expert groups, each sharded across 2 GPUs

    DeepSeek-MoE-16B on 8 GPUs:
      ep_size=8, tp_size=1  → 8 expert groups (2 experts/GPU for 16 experts)
    """
    model: str = "mixtral-8x7b"
    n_experts: int = 8
    top_k: int = 2
    n_gpus: int = 8
    ep_size: int = 8            # expert parallel degree
    tp_size: int = 1            # tensor parallel degree
    pp_size: int = 1            # pipeline parallel degree
    hidden_size: int = 4096
    intermediate_size: int = 14336
    n_layers: int = 32

    def __post_init__(self):
        assert self.ep_size * self.tp_size * self.pp_size <= self.n_gpus, \
            f"ep={self.ep_size} * tp={self.tp_size} * pp={self.pp_size} > n_gpus={self.n_gpus}"
        assert self.n_experts % self.ep_size == 0, \
            f"n_experts={self.n_experts} must be divisible by ep_size={self.ep_size}"

    @property
    def experts_per_gpu(self) -> int:
        return self.n_experts // self.ep_size

    @property
    def world_size(self) -> int:
        return self.ep_size * self.tp_size * self.pp_size

    def gpu_layout(self) -> Dict[int, Dict]:
        """Return which experts and layers each GPU handles."""
        layout = {}
        for gpu_id in range(self.n_gpus):
            ep_rank = gpu_id % self.ep_size
            expert_start = ep_rank * self.experts_per_gpu
            expert_end = expert_start + self.experts_per_gpu
            pp_rank = gpu_id // (self.ep_size * self.tp_size)
            layers_per_stage = self.n_layers // max(self.pp_size, 1)
            layer_start = pp_rank * layers_per_stage
            layout[gpu_id] = {
                "experts": list(range(expert_start, expert_end)),
                "layers": list(range(layer_start, layer_start + layers_per_stage)),
                "ep_rank": ep_rank,
                "tp_rank": (gpu_id // self.ep_size) % self.tp_size,
                "pp_rank": pp_rank,
            }
        return layout

    def print_layout(self) -> None:
        print(f"\nMoE Parallelism Layout — {self.model}")
        print(f"  n_experts={self.n_experts}  top_k={self.top_k}  n_gpus={self.n_gpus}")
        print(f"  EP={self.ep_size}  TP={self.tp_size}  PP={self.pp_size}")
        print(f"  Experts per GPU: {self.experts_per_gpu}")
        print(f"  World size: {self.world_size}")
        print()
        for gpu_id, info in self.gpu_layout().items():
            print(f"  GPU {gpu_id:2d}: experts={info['experts']}  "
                  f"layers={info['layers'][0]}..{info['layers'][-1]}  "
                  f"ep_rank={info['ep_rank']}  tp_rank={info['tp_rank']}")

    def to_vllm_args(self) -> List[str]:
        """Generate vLLM launch arguments for this parallelism config."""
        return [
            f"--tensor-parallel-size={self.tp_size}",
            f"--pipeline-parallel-size={self.pp_size}",
            f"--model={self.model}",
            f"--max-model-len=32768",
            "--enable-chunked-prefill",
            "--gpu-memory-utilization=0.92",
        ]

    def to_deepspeed_config(self) -> Dict:
        """Generate DeepSpeed-Inference config for this MoE layout."""
        return {
            "replace_with_kernel_inject": True,
            "tensor_parallel": {"tp_size": self.tp_size},
            "moe": {
                "enabled": True,
                "ep_size": self.ep_size,
                "moe_experts": self.n_experts,
                "top_k": self.top_k,
                "use_residual": False,
            },
            "dtype": "fp16",
            "enable_cuda_graph": True,
            "max_tokens": 2048,
        }


# ──────────────────────────────────────────────────────────────────────────────
# MoE load balancer
# ──────────────────────────────────────────────────────────────────────────────

class MoELoadBalancer:
    """
    Monitors expert utilisation and detects load imbalance.

    In production, expert collapse (one expert receiving >50% of tokens)
    degrades quality and creates GPU hotspots. This class:
      - Tracks rolling expert utilisation over a sliding window
      - Computes load imbalance ratio (max / mean utilisation)
      - Emits Prometheus metrics and alerts when imbalance exceeds threshold
    """

    def __init__(
        self,
        n_experts: int = 8,
        window_size: int = 1000,
        imbalance_threshold: float = 2.0,   # alert if max/mean > 2x
    ):
        self.n_experts = n_experts
        self.window_size = window_size
        self.imbalance_threshold = imbalance_threshold
        self._counts = torch.zeros(n_experts, dtype=torch.float64)
        self._total_tokens = 0
        self._step = 0

    def update(self, selected_experts: torch.Tensor) -> None:
        """
        Update utilisation counts from a routing decision.

        Args:
            selected_experts: [T, top_k] int64 — expert indices for each token
        """
        flat = selected_experts.flatten().cpu()
        for expert_id in flat.tolist():
            if 0 <= expert_id < self.n_experts:
                self._counts[expert_id] += 1
        self._total_tokens += flat.numel()
        self._step += 1

    def utilisation(self) -> torch.Tensor:
        """Return per-expert utilisation as fraction of total token-expert assignments."""
        if self._total_tokens == 0:
            return torch.zeros(self.n_experts)
        return self._counts / self._total_tokens

    def imbalance_ratio(self) -> float:
        """Max expert utilisation / mean expert utilisation."""
        util = self.utilisation()
        mean = util.mean().item()
        if mean == 0:
            return 1.0
        return (util.max() / mean).item()

    def report(self) -> Dict:
        util = self.utilisation()
        ratio = self.imbalance_ratio()
        report = {
            "step": self._step,
            "total_tokens": self._total_tokens,
            "imbalance_ratio": round(ratio, 3),
            "alert": ratio > self.imbalance_threshold,
            "expert_utilisation": {
                f"expert_{i}": f"{u*100:.1f}%" for i, u in enumerate(util.tolist())
            },
        }
        if report["alert"]:
            logger.warning(
                "MoE load imbalance detected: ratio=%.2f (threshold=%.1f). "
                "Consider increasing aux_loss_coef or using capacity_factor.",
                ratio, self.imbalance_threshold,
            )
        return report

    def push_to_prometheus(self) -> None:
        """Push expert utilisation metrics to Prometheus."""
        try:
            from observability.metrics import metrics
            if metrics is None:
                return
            util = self.utilisation()
            for i, u in enumerate(util.tolist()):
                # Reuse queue_depth gauge with expert label
                metrics.queue_depth.labels(queue_name=f"moe_expert_{i}").set(u * 100)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# MoE architecture config + named presets
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MoEArchConfig:
    """
    Structured architecture metadata for a Mixture-of-Experts model.

    Decouples architecture facts (param counts, head counts) from serving
    decisions so that memory estimation and parallelism advice can be
    computed without loading the model.
    """
    model_id: str
    num_experts: int
    top_k_experts: int
    num_layers: int
    hidden_size: int
    intermediate_size: int      # per-expert FFN intermediate width
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    max_position_embeddings: int
    total_params_b: float       # total parameters (billions)
    active_params_b: float      # active params per forward pass (billions)
    expert_type: str = "dense"  # dense | sparse | shared

    def efficiency_ratio(self) -> float:
        """active_params / total_params — higher means more efficient MoE."""
        return self.active_params_b / self.total_params_b

    def __str__(self) -> str:
        return (
            f"{self.model_id}  "
            f"({self.num_experts}x experts, top-{self.top_k_experts})  "
            f"{self.active_params_b:.1f}B active / {self.total_params_b:.1f}B total  "
            f"efficiency={self.efficiency_ratio():.1%}"
        )


# Named architecture presets
MIXTRAL_8X7B = MoEArchConfig(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    num_experts=8, top_k_experts=2,
    num_layers=32, hidden_size=4096, intermediate_size=14336,
    num_attention_heads=32, num_kv_heads=8,
    vocab_size=32000, max_position_embeddings=32768,
    total_params_b=46.7, active_params_b=12.9,
)

MIXTRAL_8X22B = MoEArchConfig(
    model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    num_experts=8, top_k_experts=2,
    num_layers=56, hidden_size=6144, intermediate_size=16384,
    num_attention_heads=48, num_kv_heads=8,
    vocab_size=32000, max_position_embeddings=65536,
    total_params_b=141.0, active_params_b=39.1,
)

DEEPSEEK_V2 = MoEArchConfig(
    model_id="deepseek-ai/DeepSeek-V2",
    num_experts=160, top_k_experts=6,
    num_layers=60, hidden_size=5120, intermediate_size=1536,
    num_attention_heads=128, num_kv_heads=128,
    vocab_size=102400, max_position_embeddings=163840,
    total_params_b=236.0, active_params_b=21.0,
    expert_type="sparse",
)

ARCH_PRESETS: Dict[str, MoEArchConfig] = {
    "mixtral-8x7b": MIXTRAL_8X7B,
    "mixtral-8x22b": MIXTRAL_8X22B,
    "deepseek-v2": DEEPSEEK_V2,
}


# ──────────────────────────────────────────────────────────────────────────────
# MoE layer (router + expert pool) — runnable nn.Module
# ──────────────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Single MoE FFN layer: replaces the standard dense FFN in a transformer
    with N expert FFNs gated by a learned router.

    Each token is routed to top_k experts; their outputs are weighted-summed.
    The auxiliary load-balancing loss is returned alongside the output so the
    caller can add it to the main loss during training.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        routing_type: str = "softmax",
    ):
        super().__init__()
        self.router = MoERouter(
            hidden_size=hidden_size,
            n_experts=num_experts,
            top_k=top_k,
            routing_type=routing_type,
        )
        # SwiGLU-style experts (gate proj + up proj + down proj)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch * seq_len, hidden_size]  (caller must flatten batch/seq dims)

        Returns:
            output:   [batch * seq_len, hidden_size]
            aux_loss: scalar load-balancing loss (None during eval)
        """
        routing_weights, selected_experts, _, aux_loss = self.router(x)

        output = torch.zeros_like(x)
        for k in range(self.router.top_k):
            expert_idx = selected_experts[:, k]        # (T,)
            weights = routing_weights[:, k].unsqueeze(-1)  # (T, 1)
            for e_id, expert in enumerate(self.experts):
                mask = expert_idx == e_id
                if mask.any():
                    output[mask] += weights[mask] * expert(x[mask])

        return output, aux_loss


# ──────────────────────────────────────────────────────────────────────────────
# High-level vLLM serving config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MoEServingConfig:
    """
    High-level vLLM serving configuration for MoE models.

    Provides:
      - Factory methods for common models (for_mixtral, for_deepseek_v2)
      - vLLM launch command generation
      - Ray Serve YAML config generation
      - GPU memory estimation (weights + KV cache)
      - Docker Compose generation

    Example:
        cfg = MoEServingConfig.for_mixtral(n_gpus=4)
        print(cfg.generate_vllm_launch_cmd())
        print(cfg.estimate_gpu_memory_gb())
    """
    arch: MoEArchConfig = field(default_factory=lambda: MIXTRAL_8X7B)
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    quantization: Optional[str] = None     # None | awq | gptq | fp8
    kv_cache_dtype: str = "auto"
    use_speculative: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5

    @property
    def world_size(self) -> int:
        return self.tp_size * self.ep_size * self.pp_size

    @classmethod
    def for_mixtral(cls, n_gpus: int = 4) -> "MoEServingConfig":
        """Mixtral-8x7B: tensor-parallel across all GPUs."""
        return cls(arch=MIXTRAL_8X7B, tp_size=n_gpus, ep_size=1, pp_size=1)

    @classmethod
    def for_mixtral_22b(cls, n_gpus: int = 8) -> "MoEServingConfig":
        """Mixtral-8x22B: TP=4, PP=2 for 8-GPU nodes."""
        return cls(arch=MIXTRAL_8X22B, tp_size=4, ep_size=1, pp_size=n_gpus // 4)

    @classmethod
    def for_deepseek_v2(cls, n_gpus: int = 8) -> "MoEServingConfig":
        """DeepSeek-V2: expert-parallel (160 experts → 8 EP groups)."""
        return cls(arch=DEEPSEEK_V2, tp_size=1, ep_size=n_gpus, pp_size=1)

    def generate_vllm_launch_cmd(self) -> str:
        """Generate the `python -m vllm.entrypoints.openai.api_server` command."""
        parts = [
            "python -m vllm.entrypoints.openai.api_server",
            f"--model {self.arch.model_id}",
            f"--tensor-parallel-size {self.tp_size}",
            f"--pipeline-parallel-size {self.pp_size}",
            f"--max-model-len {self.max_model_len}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--max-num-seqs {self.max_num_seqs}",
            f"--max-num-batched-tokens {self.max_num_batched_tokens}",
            "--trust-remote-code",
            "--served-model-name llmops-moe",
        ]
        if self.ep_size > 1:
            parts.append("--enable-expert-parallel")
        if self.enable_chunked_prefill:
            parts.append("--enable-chunked-prefill")
        if self.quantization:
            parts.append(f"--quantization {self.quantization}")
        if self.use_speculative and self.speculative_model:
            parts += [
                f"--speculative-model {self.speculative_model}",
                f"--num-speculative-tokens {self.num_speculative_tokens}",
            ]
        return " \\\n  ".join(parts)

    def generate_ray_serve_config(self, deployment_name: str = "moe-llm") -> Dict:
        """Generate Ray Serve deployment config dict (serialisable to YAML)."""
        return {
            "applications": [{
                "name": deployment_name,
                "route_prefix": "/",
                "import_path": "vllm.entrypoints.openai.api_server:build_app",
                "args": {
                    "model": self.arch.model_id,
                    "tensor_parallel_size": self.tp_size,
                    "pipeline_parallel_size": self.pp_size,
                    "max_model_len": self.max_model_len,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "trust_remote_code": True,
                },
                "deployments": [{
                    "name": "VLLMDeployment",
                    "num_replicas": 1,
                    "ray_actor_options": {"num_gpus": self.world_size},
                    "max_concurrent_queries": self.max_num_seqs,
                }],
            }]
        }

    def estimate_gpu_memory_gb(self) -> Dict:
        """
        Estimate GPU memory requirements.

        Accounts for model weights (dtype-dependent) and KV cache
        (proportional to max_model_len × max_num_seqs × layer/head dims).
        """
        bytes_per_param = {
            "float16": 2, "bfloat16": 2,
            "fp8": 1, "awq": 0.5, "gptq": 0.5, "int4": 0.5,
        }
        bpp = bytes_per_param.get(self.quantization or "float16", 2)
        model_gb = self.arch.total_params_b * 1e9 * bpp / (1024 ** 3)

        head_dim = self.arch.hidden_size // self.arch.num_attention_heads
        kv_gb = (
            2                               # K and V
            * self.arch.num_layers
            * self.arch.num_kv_heads
            * head_dim
            * self.max_model_len
            * self.max_num_seqs
            * 2                             # fp16 KV cache
            / (1024 ** 3)
        )
        total_gb = model_gb + kv_gb
        per_gpu_gb = total_gb / self.world_size

        return {
            "model_weights_gb": round(model_gb, 1),
            "kv_cache_gb": round(kv_gb, 1),
            "total_gb": round(total_gb, 1),
            "per_gpu_gb": round(per_gpu_gb, 1),
            "world_size": self.world_size,
            "recommended_gpu": "A100-80GB" if per_gpu_gb < 80 else "H100-80GB",
        }

    def generate_docker_compose(self) -> str:
        """Generate a Docker Compose service definition for this config."""
        cmd = self.generate_vllm_launch_cmd()
        return f"""\
# MoE Model Serving — Docker Compose
# Model:   {self.arch.model_id}
# GPUs:    {self.world_size}x  (TP={self.tp_size} EP={self.ep_size} PP={self.pp_size})
# Memory:  ~{self.estimate_gpu_memory_gb()['per_gpu_gb']} GB/GPU

services:
  moe-server:
    image: vllm/vllm-openai:latest
    command: >
      {cmd}
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${{HF_TOKEN}}
      - NCCL_DEBUG=INFO
      - NCCL_SOCKET_IFNAME=eth0
    volumes:
      - hf-cache:/root/.cache/huggingface
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {self.world_size}
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 5

volumes:
  hf-cache:
"""


# ──────────────────────────────────────────────────────────────────────────────
# DeepSpeed-Inference MoE config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepSpeedMoEConfig:
    """
    Full DeepSpeed-Inference configuration for MoE model serving.

    Covers:
      - Kernel injection (fused attention, MLP)
      - Expert parallelism via DeepSpeed EP groups
      - INT8 quantization (SmoothQuant)
      - CUDA graph capture
      - Dynamic batching
    """
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ep_config: ExpertParallelConfig = field(default_factory=ExpertParallelConfig)
    dtype: str = "fp16"                 # fp16 | bf16 | int8
    max_tokens: int = 4096
    enable_cuda_graph: bool = True
    cuda_graph_max_batch_size: int = 32
    quantization: str = "none"          # none | int8 | int4

    def to_dict(self) -> Dict:
        cfg = {
            "replace_with_kernel_inject": True,
            "dtype": self.dtype,
            "enable_cuda_graph": self.enable_cuda_graph,
            "tensor_parallel": {"tp_size": self.ep_config.tp_size},
            "moe": {
                "enabled": True,
                "ep_size": self.ep_config.ep_size,
                "moe_experts": self.ep_config.n_experts,
                "top_k": self.ep_config.top_k,
            },
            "max_tokens": self.max_tokens,
        }
        if self.quantization == "int8":
            cfg["quantization"] = {
                "enabled": True,
                "quantize_verbose": False,
                "kernel_inject": True,
                "moe_quantization": True,
                "quantize_weight": {"enabled": True, "quantized_initialization": {"num_bits": 8}},
            }
        return cfg

    def save(self, path: str = "configs/deepspeed_moe.json") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("DeepSpeed MoE config saved: %s", path)

    def load_model(self):
        """Load and optimise a MoE model with DeepSpeed-Inference."""
        try:
            import deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading %s with DeepSpeed-Inference ...", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.dtype == "fp16" else torch.bfloat16,
                device_map="auto",
            )
            ds_model = deepspeed.init_inference(model, config=self.to_dict())
            logger.info("DeepSpeed-Inference initialised")
            return ds_model, tokenizer
        except ImportError:
            raise ImportError("deepspeed required: pip install deepspeed")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU inference config (transformers device_map)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MultiGPUInferenceConfig:
    """
    Multi-GPU inference using transformers device_map + pipeline parallelism.
    Works without DeepSpeed — uses HuggingFace Accelerate for layer distribution.

    Strategies:
      "auto"       — Accelerate distributes layers evenly across available GPUs
      "balanced"   — Balances memory usage across GPUs
      "sequential" — Layers assigned sequentially (pipeline parallelism)
      custom dict  — Explicit layer-to-device mapping
    """
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    device_map: str = "auto"            # "auto" | "balanced" | "sequential" | dict
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_memory: Optional[Dict[int, str]] = None   # e.g. {0: "20GiB", 1: "20GiB"}
    offload_folder: Optional[str] = None          # CPU offload for very large models

    def load(self):
        """Load model with the configured multi-GPU strategy."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading %s with device_map=%s", self.model_name, self.device_map)

        kwargs = {
            "device_map": self.device_map,
            "torch_dtype": getattr(torch, self.dtype),
        }
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if self.offload_folder:
            kwargs["offload_folder"] = self.offload_folder

        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        # Log actual device assignment
        if hasattr(model, "hf_device_map"):
            device_counts: Dict[str, int] = {}
            for dev in model.hf_device_map.values():
                device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
            logger.info("Device map: %s", device_counts)

        return model, tokenizer

    def generate_device_map(self, n_gpus: int, n_layers: int) -> Dict[str, str]:
        """
        Generate a balanced manual device map for pipeline parallelism.
        Assigns equal numbers of transformer layers to each GPU.
        """
        layers_per_gpu = n_layers // n_gpus
        device_map: Dict[str, str] = {
            "model.embed_tokens": "cuda:0",
            "model.norm": f"cuda:{n_gpus - 1}",
            "lm_head": f"cuda:{n_gpus - 1}",
        }
        for layer_idx in range(n_layers):
            gpu_id = min(layer_idx // layers_per_gpu, n_gpus - 1)
            device_map[f"model.layers.{layer_idx}"] = f"cuda:{gpu_id}"
        return device_map


# ──────────────────────────────────────────────────────────────────────────────
# Ray Serve deployment
# ──────────────────────────────────────────────────────────────────────────────

def create_ray_serve_moe_deployment(
    ep_config: ExpertParallelConfig,
    num_replicas: int = 2,
    ray_actor_options: Optional[Dict] = None,
):
    """
    Create a Ray Serve deployment for MoE model serving.

    Each replica runs on a separate node with the configured GPU allocation.
    Ray handles load balancing across replicas.

    Requires: ray[serve] + vLLM or DeepSpeed
    """
    try:
        from ray import serve
        from ray.serve.handle import DeploymentHandle
    except ImportError:
        raise ImportError("ray[serve] required: pip install 'ray[serve]'")

    if ray_actor_options is None:
        ray_actor_options = {
            "num_gpus": ep_config.n_gpus / num_replicas,
        }

    @serve.deployment(
        name=f"moe-{ep_config.model.replace('/', '-')}",
        num_replicas=num_replicas,
        ray_actor_options=ray_actor_options,
        max_ongoing_requests=64,
        health_check_period_s=30,
        health_check_timeout_s=60,
    )
    class MoEDeployment:
        def __init__(self):
            self.config = ep_config
            self._model = None
            self._tokenizer = None
            logger.info("MoE deployment initialising: %s", ep_config.model)

        def _load_model(self):
            """Lazy model loading — called on first request."""
            if self._model is None:
                multi_gpu_cfg = MultiGPUInferenceConfig(
                    model_name=ep_config.model,
                    device_map="auto",
                    dtype="float16",
                )
                self._model, self._tokenizer = multi_gpu_cfg.load()
                logger.info("Model loaded on replica")

        async def __call__(self, request) -> Dict:
            self._load_model()
            data = await request.json()
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 512)

            inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"text": text, "latency_ms": latency_ms, "model": ep_config.model}

        def check_health(self):
            return {"status": "ok", "model": ep_config.model}

    return MoEDeployment


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE serving utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Show parallelism layout
    layout_p = subparsers.add_parser("layout", help="Print GPU layout for a MoE config")
    layout_p.add_argument("--model", default="mixtral-8x7b")
    layout_p.add_argument("--n-experts", type=int, default=8)
    layout_p.add_argument("--n-gpus", type=int, default=8)
    layout_p.add_argument("--ep-size", type=int, default=8)
    layout_p.add_argument("--tp-size", type=int, default=1)

    # Save DeepSpeed config
    ds_p = subparsers.add_parser("deepspeed-config", help="Generate DeepSpeed MoE config")
    ds_p.add_argument("--output", default="configs/deepspeed_moe.json")
    ds_p.add_argument("--ep-size", type=int, default=8)
    ds_p.add_argument("--dtype", default="fp16")

    # Router demo (CPU)
    router_p = subparsers.add_parser("router-demo", help="Demo MoE router on CPU")
    router_p.add_argument("--n-experts", type=int, default=8)
    router_p.add_argument("--top-k", type=int, default=2)
    router_p.add_argument("--batch", type=int, default=32)
    router_p.add_argument("--seq-len", type=int, default=128)

    # High-level serving config
    serve_p = subparsers.add_parser("serve-config", help="Show vLLM launch command + memory estimate")
    serve_p.add_argument("--model", default="mixtral-8x7b",
                         choices=list(ARCH_PRESETS.keys()))
    serve_p.add_argument("--n-gpus", type=int, default=4)
    serve_p.add_argument("--gen-compose", action="store_true",
                         help="Also print Docker Compose YAML")
    serve_p.add_argument("--gen-ray", action="store_true",
                         help="Also print Ray Serve config JSON")

    # MoE layer forward-pass demo
    layer_demo_p = subparsers.add_parser("layer-demo", help="Run MoE layer forward pass")
    layer_demo_p.add_argument("--n-experts", type=int, default=8)
    layer_demo_p.add_argument("--top-k", type=int, default=2)
    layer_demo_p.add_argument("--batch", type=int, default=4)
    layer_demo_p.add_argument("--seq-len", type=int, default=16)
    layer_demo_p.add_argument("--hidden", type=int, default=256)

    args = parser.parse_args()

    if args.command == "layout":
        cfg = ExpertParallelConfig(
            model=args.model,
            n_experts=args.n_experts,
            n_gpus=args.n_gpus,
            ep_size=args.ep_size,
            tp_size=args.tp_size,
        )
        cfg.print_layout()
        print("\nvLLM args:", " ".join(cfg.to_vllm_args()))

    elif args.command == "deepspeed-config":
        ep_cfg = ExpertParallelConfig(ep_size=args.ep_size)
        ds_cfg = DeepSpeedMoEConfig(ep_config=ep_cfg, dtype=args.dtype)
        ds_cfg.save(args.output)
        print(f"DeepSpeed config saved: {args.output}")
        print(json.dumps(ds_cfg.to_dict(), indent=2))

    elif args.command == "router-demo":
        print(f"\nMoE Router demo — n_experts={args.n_experts} top_k={args.top_k}")
        router = MoERouter(
            hidden_size=512,
            n_experts=args.n_experts,
            top_k=args.top_k,
        )
        hidden = torch.randn(args.batch * args.seq_len, 512)
        weights, experts, logits, _ = router(hidden)
        print(f"Input shape: {hidden.shape}")
        print(f"Routing weights: {weights.shape}  (sum={weights.sum(-1).mean():.3f})")
        print(f"Selected experts: {experts.shape}")
        stats = router.get_routing_stats(logits)
        print("Expert utilisation:", stats)

        balancer = MoELoadBalancer(n_experts=args.n_experts)
        balancer.update(experts)
        report = balancer.report()
        print(f"Load imbalance ratio: {report['imbalance_ratio']:.2f}x")
        print(f"Alert: {report['alert']}")

    elif args.command == "serve-config":
        arch = ARCH_PRESETS[args.model]
        if args.model == "mixtral-8x7b":
            cfg = MoEServingConfig.for_mixtral(n_gpus=args.n_gpus)
        elif args.model == "mixtral-8x22b":
            cfg = MoEServingConfig.for_mixtral_22b(n_gpus=args.n_gpus)
        else:
            cfg = MoEServingConfig.for_deepseek_v2(n_gpus=args.n_gpus)

        print(f"\n{'='*60}")
        print(f"Model: {arch}")
        print(f"{'='*60}\n")
        print("vLLM launch command:")
        print(cfg.generate_vllm_launch_cmd())
        print("\nMemory estimate:")
        print(json.dumps(cfg.estimate_gpu_memory_gb(), indent=2))

        if args.gen_compose:
            print("\nDocker Compose:")
            print(cfg.generate_docker_compose())

        if args.gen_ray:
            print("\nRay Serve config:")
            print(json.dumps(cfg.generate_ray_serve_config(), indent=2))

    elif args.command == "layer-demo":
        print(f"\nMoE Layer demo — n_experts={args.n_experts} top_k={args.top_k}")
        layer = MoELayer(
            hidden_size=args.hidden,
            intermediate_size=args.hidden * 2,
            num_experts=args.n_experts,
            top_k=args.top_k,
        )
        x = torch.randn(args.batch * args.seq_len, args.hidden)
        out, aux_loss = layer(x)
        print(f"Input:  {x.shape}")
        print(f"Output: {out.shape}")
        print(f"Aux loss: {aux_loss.item() if aux_loss is not None else 'N/A (eval mode)'}")
        stats = layer.router.get_routing_stats(layer.router.gate(x))
        print("Expert utilisation:", stats)

    else:
        parser.print_help()
