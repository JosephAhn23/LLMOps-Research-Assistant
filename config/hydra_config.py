"""
Hydra-based hierarchical configuration system for LLMOps Research Assistant.

Provides:
  - Structured configs with dataclass validation
  - Environment scoping (local / dev / staging / prod)
  - Experiment override composition
  - Config serialisation for reproducibility
  - CLI override support (dot-notation: key=value)

Config hierarchy:
  conf/
    config.yaml              <- root (assembles all groups)
    model/
      llama3_1b.yaml
      llama3_8b.yaml
      gpt4o_mini.yaml
    retrieval/
      faiss.yaml
      hybrid.yaml
      azure_search.yaml
    training/
      qlora.yaml
      rlhf_ppo.yaml
    eval/
      ragas.yaml
      deepeval.yaml
    infra/
      local.yaml
      aws.yaml
      azure.yaml
    experiment/
      baseline.yaml
      ablation_retrieval.yaml
      ablation_reranking.yaml

Usage:
    # Python
    cfg = LLMOpsConfig.load("conf/config.yaml",
                            overrides=["model=llama3_8b", "+experiment=ablation_retrieval"])

    # CLI (via Hydra decorator)
    python train.py model=llama3_8b retrieval=hybrid training.learning_rate=1e-4

    # Environment scoping
    LLMOPS_ENV=prod python train.py
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Structured config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    backend: str = "vllm"               # vllm | llamacpp | openai | azure_openai
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    quantization: Optional[str] = None  # None | qlora | awq | gptq
    n_gpu_layers: int = 0               # llamacpp only
    dtype: str = "auto"                 # auto | float16 | bfloat16 | float32
    trust_remote_code: bool = False


@dataclass
class RetrievalConfig:
    backend: str = "hybrid"             # faiss | hybrid | azure_search
    top_k: int = 10
    bm25_candidates: int = 100
    faiss_candidates: int = 100
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss.index"
    azure_search_endpoint: str = ""
    azure_search_index: str = "llmops-docs"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class TrainingConfig:
    method: str = "qlora"               # qlora | rlhf_ppo | sft | dpo
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "models/finetuned"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    # Training loop
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_seq_length: int = 512
    # RLHF
    reward_model: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    ppo_epochs: int = 1
    kl_penalty: float = 0.1


@dataclass
class EvalConfig:
    frameworks: List[str] = field(
        default_factory=lambda: ["ragas", "deepeval"]
    )
    ragas_metrics: List[str] = field(
        default_factory=lambda: [
            "faithfulness", "answer_relevancy", "context_precision"
        ]
    )
    deepeval_metrics: List[str] = field(
        default_factory=lambda: ["relevancy", "faithfulness", "hallucination"]
    )
    faithfulness_min: float = 0.70
    relevancy_min: float = 0.70
    hallucination_min: float = 0.80
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    regression_threshold: float = 0.05
    baseline_path: str = "mlops/baselines/"


@dataclass
class InfraConfig:
    environment: str = "local"          # local | dev | staging | prod
    cloud: str = "aws"                  # aws | azure | gcp | local
    # AWS
    aws_region: str = "us-east-1"
    s3_bucket: str = "llmops-artifacts"
    sagemaker_role: str = ""
    # Azure
    azure_subscription_id: str = ""
    azure_resource_group: str = "llmops-rg"
    azure_search_endpoint: str = ""
    # Kubernetes
    k8s_namespace: str = "llmops"
    k8s_replicas: int = 2
    # Streaming
    kafka_bootstrap: str = "localhost:9092"
    kinesis_region: str = "us-east-1"
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "llmops-main"


@dataclass
class ExperimentConfig:
    """Experiment-level overrides — compose on top of the base config."""
    name: str = "baseline"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    ablate_reranking: bool = False
    ablate_bm25: bool = False
    ablate_faiss: bool = False
    seed: int = 42
    log_every_n_steps: int = 10
    save_every_n_steps: int = 100


@dataclass
class LLMOpsConfig:
    """Root config — assembles all config groups."""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_path: str = "conf/config.yaml",
        overrides: Optional[List[str]] = None,
    ) -> "LLMOpsConfig":
        """
        Load config from YAML, then apply dot-notation overrides.
        Override syntax: "model.temperature=0.5" or "retrieval=hybrid"
        """
        import yaml

        cfg = cls()
        if os.path.exists(config_path):
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            cfg = cls._from_dict(raw)
            logger.info("Config loaded from %s", config_path)
        else:
            logger.info("Config file not found — using defaults (%s)", config_path)

        if overrides:
            cfg = cfg._apply_overrides(overrides)

        return cfg

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "LLMOpsConfig":
        def _safe(klass, data):
            if not isinstance(data, dict):
                return klass()
            valid = {f.name for f in klass.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            return klass(**{k: v for k, v in data.items() if k in valid})

        return cls(
            model=_safe(ModelConfig, d.get("model", {})),
            retrieval=_safe(RetrievalConfig, d.get("retrieval", {})),
            training=_safe(TrainingConfig, d.get("training", {})),
            eval=_safe(EvalConfig, d.get("eval", {})),
            infra=_safe(InfraConfig, d.get("infra", {})),
            experiment=_safe(ExperimentConfig, d.get("experiment", {})),
        )

    def _apply_overrides(self, overrides: List[str]) -> "LLMOpsConfig":
        """Apply dot-notation overrides in place on a deep copy."""
        import copy

        cfg = copy.deepcopy(self)
        for override in overrides:
            if "=" not in override:
                continue
            key_path, value = override.split("=", 1)
            parts = key_path.lstrip("+").split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    logger.warning("Override path not found: %s", key_path)
                    break
            if obj is None:
                continue
            field_name = parts[-1]
            current = getattr(obj, field_name, None)
            try:
                if isinstance(current, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, list):
                    value = json.loads(value)
                setattr(obj, field_name, value)
            except (ValueError, json.JSONDecodeError):
                setattr(obj, field_name, value)
        return cfg

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses

        def _convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _convert(self)

    def save(self, path: str) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved: %s", path)

    # ── MLflow ───────────────────────────────────────────────────────────────

    def log_to_mlflow(self) -> None:
        """Log all config values as MLflow params (flattened dot-notation keys)."""
        try:
            import mlflow

            mlflow.log_params(self._flatten(self.to_dict()))
        except Exception as e:
            logger.warning("MLflow param logging failed: %s", e)

    def _flatten(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten(v, key))
            elif isinstance(v, list):
                result[key] = str(v)
            else:
                result[key] = v
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Hydra decorator
# ──────────────────────────────────────────────────────────────────────────────

def hydra_main(cfg_path: str = "conf"):
    """
    Decorator factory for Hydra-powered entry points.
    Falls back to manual YAML loading when Hydra is not installed.

    Usage:
        @hydra_main("conf")
        def main(cfg: LLMOpsConfig):
            print(cfg.model.name)

        main()
    """
    def decorator(fn):
        try:
            import hydra
            from omegaconf import DictConfig, OmegaConf

            @hydra.main(version_base=None, config_path=cfg_path, config_name="config")
            def _wrapped(hydra_cfg: DictConfig):
                raw = OmegaConf.to_container(hydra_cfg, resolve=True)
                cfg = LLMOpsConfig._from_dict(raw)
                return fn(cfg)

            return _wrapped

        except ImportError:
            logger.warning("Hydra not installed — using manual YAML loader.")

            def _fallback(**kwargs):
                cfg = LLMOpsConfig.load(
                    f"{cfg_path}/config.yaml",
                    overrides=kwargs.get("overrides", []),
                )
                return fn(cfg)

            return _fallback

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Config file generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_config_files(output_dir: str = "conf") -> List[str]:
    """
    Write the full Hydra conf/ directory structure to disk.
    Safe to re-run — overwrites existing files.
    """
    import yaml

    files: Dict[str, Any] = {
        "config.yaml": {
            "defaults": [
                {"model": "gpt4o_mini"},
                {"retrieval": "hybrid"},
                {"training": "qlora"},
                {"eval": "ragas"},
                {"infra": "local"},
                "_self_",
            ],
            "experiment": {"name": "baseline", "seed": 42},
        },
        # ── model ──────────────────────────────────────────────────────────
        "model/gpt4o_mini.yaml": {
            "name": "gpt-4o-mini",
            "backend": "openai",
            "max_tokens": 512,
            "temperature": 0.0,
        },
        "model/llama3_1b.yaml": {
            "name": "meta-llama/Llama-3.2-1B-Instruct",
            "backend": "llamacpp",
            "max_tokens": 512,
            "temperature": 0.0,
            "n_gpu_layers": 32,
        },
        "model/llama3_8b.yaml": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "backend": "vllm",
            "max_tokens": 1024,
            "temperature": 0.0,
            "quantization": "awq",
        },
        # ── retrieval ──────────────────────────────────────────────────────
        "retrieval/faiss.yaml": {
            "backend": "faiss",
            "top_k": 10,
            "rerank": False,
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "retrieval/hybrid.yaml": {
            "backend": "hybrid",
            "top_k": 10,
            "bm25_candidates": 100,
            "faiss_candidates": 100,
            "rerank": True,
        },
        "retrieval/azure_search.yaml": {
            "backend": "azure_search",
            "top_k": 10,
            "rerank": True,
            "azure_search_index": "llmops-docs",
        },
        # ── training ───────────────────────────────────────────────────────
        "training/qlora.yaml": {
            "method": "qlora",
            "lora_r": 16,
            "lora_alpha": 32,
            "load_in_4bit": True,
            "num_epochs": 3,
            "learning_rate": 2e-4,
        },
        "training/rlhf_ppo.yaml": {
            "method": "rlhf_ppo",
            "ppo_epochs": 1,
            "kl_penalty": 0.1,
            "learning_rate": 1e-5,
        },
        "training/dpo.yaml": {
            "method": "dpo",
            "num_epochs": 1,
            "learning_rate": 5e-5,
            "beta": 0.1,
        },
        # ── eval ───────────────────────────────────────────────────────────
        "eval/ragas.yaml": {
            "frameworks": ["ragas"],
            "faithfulness_min": 0.70,
            "relevancy_min": 0.70,
            "regression_threshold": 0.05,
        },
        "eval/deepeval.yaml": {
            "frameworks": ["deepeval"],
            "faithfulness_min": 0.70,
            "hallucination_min": 0.80,
        },
        # ── infra ──────────────────────────────────────────────────────────
        "infra/local.yaml": {
            "environment": "local",
            "cloud": "local",
            "mlflow_tracking_uri": "http://localhost:5000",
            "kafka_bootstrap": "localhost:9092",
        },
        "infra/aws.yaml": {
            "environment": "prod",
            "cloud": "aws",
            "aws_region": "us-east-1",
            "mlflow_tracking_uri": "${oc.env:MLFLOW_TRACKING_URI}",
            "kafka_bootstrap": "${oc.env:KAFKA_BOOTSTRAP_SERVERS}",
        },
        "infra/azure.yaml": {
            "environment": "prod",
            "cloud": "azure",
            "mlflow_tracking_uri": "${oc.env:MLFLOW_TRACKING_URI}",
            "azure_subscription_id": "${oc.env:AZURE_SUBSCRIPTION_ID}",
            "azure_resource_group": "${oc.env:AZURE_RESOURCE_GROUP,llmops-rg}",
        },
        # ── experiment ─────────────────────────────────────────────────────
        "experiment/baseline.yaml": {
            "name": "baseline",
            "description": "Full pipeline with hybrid retrieval and reranking",
            "seed": 42,
        },
        "experiment/ablation_retrieval.yaml": {
            "name": "ablation-retrieval",
            "description": "FAISS-only retrieval, no BM25",
            "ablate_bm25": True,
            "seed": 42,
        },
        "experiment/ablation_reranking.yaml": {
            "name": "ablation-reranking",
            "description": "Hybrid retrieval without cross-encoder reranking",
            "ablate_reranking": True,
            "seed": 42,
        },
    }

    for rel_path, content in files.items():
        full_path = Path(output_dir) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated %d config files in %s/", len(files), output_dir)
    return list(files.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Environment-scoped loader
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentConfig:
    """
    Load the config appropriate for the current deployment environment.
    Reads LLMOPS_ENV=local|dev|staging|prod (defaults to local).
    """

    ENV_OVERRIDES: Dict[str, List[str]] = {
        "local":   ["infra=local"],
        "dev":     ["infra=local", "model=gpt4o_mini",
                    "+experiment.tags.env=dev"],
        "staging": ["infra=aws", "model=llama3_8b",
                    "+experiment.tags.env=staging"],
        "prod":    ["infra=aws", "model=llama3_8b",
                    "+experiment.tags.env=prod",
                    "eval.faithfulness_min=0.75"],
    }

    @classmethod
    def load(cls, base_config: str = "conf/config.yaml") -> LLMOpsConfig:
        env = os.environ.get("LLMOPS_ENV", "local")
        overrides = cls.ENV_OVERRIDES.get(env, [])
        logger.info("Environment: %s | overrides: %s", env, overrides)
        return LLMOpsConfig.load(base_config, overrides=overrides)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hydra config utilities")
    parser.add_argument("--generate", action="store_true",
                        help="Write conf/ directory to disk")
    parser.add_argument("--conf-dir", default="conf")
    parser.add_argument("--show", action="store_true",
                        help="Print resolved config as YAML")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    if args.generate:
        generated = generate_config_files(args.conf_dir)
        print(f"Generated {len(generated)} files in {args.conf_dir}/")

    if args.show:
        cfg = LLMOpsConfig.load(
            f"{args.conf_dir}/config.yaml",
            overrides=args.overrides,
        )
        import yaml
        print(yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))
