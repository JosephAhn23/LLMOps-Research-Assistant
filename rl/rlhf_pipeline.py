"""
RLHF Pipeline with PPO for LLM Fine-Tuning.

Covers: RLHF, PPO, reward modeling, agentic training, KL-penalised policy
        optimisation, TRL integration, MLflow experiment tracking.

Architecture:
  1. AutoModelForCausalLMWithValueHead — policy with built-in value head (TRL)
  2. create_reference_model            — frozen KL reference copy
  3. RewardModel                       — HuggingFace sequence-classifier reward scorer
  4. RLHFTrainer                       — full PPO training loop with MLflow tracking
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from mlops.compat import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLHFConfig:
    # Model
    base_model: str = "Qwen/Qwen2-0.5B-Instruct"
    reward_model: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    # PPO hyperparams
    learning_rate: float = 1.41e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_steps: int = 200
    clip_range: float = 0.2
    vf_coef: float = 0.1
    kl_penalty: str = "kl"      # 'kl' | 'abs' | 'mse' | 'full'
    target_kl: float = 6.0
    init_kl_coef: float = 0.2

    # Generation
    max_new_tokens: int = 128
    min_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"

    # MLflow / output
    experiment_name: str = "rlhf-ppo"
    output_dir: str = "./outputs/rlhf"


# ---------------------------------------------------------------------------
# Reward Model Wrapper
# ---------------------------------------------------------------------------

class RewardModel:
    """
    Wraps a HuggingFace reward model (sequence classifier returning scalar reward).
    Compatible with OpenAssistant and similar reward models.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        logger.info("Loading reward model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, response: str) -> float:
        """Return scalar reward for a query-response pair."""
        text = f"{query}\n{response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        logits = self.model(**inputs).logits
        # Most reward models output a single logit (higher = better)
        return logits[0, 0].item()

    def batch_score(self, queries: List[str], responses: List[str]) -> List[float]:
        return [self.score(q, r) for q, r in zip(queries, responses)]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_ppo_dataset(tokenizer, prompts: List[str], max_length: int = 256) -> Dataset:
    """Tokenize prompts for PPO training."""

    def tokenize(sample):
        ids = tokenizer.encode(sample["query"], truncation=True, max_length=max_length)
        sample["input_ids"] = ids
        return sample

    dataset = Dataset.from_dict({"query": prompts})
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    return dataset


def load_prompts_from_hf(dataset_name: str, n: int = 1000) -> List[str]:
    """Load and extract prompts from a HuggingFace preference dataset."""
    try:
        ds = load_dataset(dataset_name, split="train")
        ds = ds.select(range(min(n, len(ds))))
        prompts = []
        for row in ds:
            text = row.get("chosen", "")
            if "\nAssistant:" in text:
                prompt = text.split("\nAssistant:")[0] + "\nAssistant:"
            else:
                prompt = text[:256]
            prompts.append(prompt)
        return prompts
    except Exception:
        logger.warning("Could not load %s — using synthetic prompts", dataset_name)
        return [
            "Summarize the key findings of this research paper:",
            "Explain the concept of attention in transformers:",
            "What are the trade-offs between RAG and fine-tuning?",
            "Describe how RLHF improves LLM alignment:",
            "Write a concise technical explanation of FAISS:",
        ] * 20


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RLHFTrainer:
    """Full RLHF-PPO training loop using TRL with a frozen reference model."""

    def __init__(self, config: RLHFConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        logger.info("Loading policy model: %s", config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy (actor + value head) and frozen reference model
        self._trl_available = self._check_trl()
        if self._trl_available:
            from trl import AutoModelForCausalLMWithValueHead, create_reference_model
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                config.base_model, torch_dtype=torch.float32
            )
            self.ref_model = create_reference_model(self.model)
        else:
            logger.warning("TRL not available — using plain causal LM with hand-rolled fallback")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model, torch_dtype=torch.float32
            ).to(self.device)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.base_model, torch_dtype=torch.float32
            ).to(self.device)
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self.reward_model = RewardModel(config.reward_model, device=self.device)

        self.gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "min_new_tokens": config.min_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def _check_trl(self) -> bool:
        try:
            from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_ppo_trainer(self):
        from trl import PPOConfig, PPOTrainer

        ppo_config = PPOConfig(
            model_name=self.config.base_model,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            max_grad_norm=1.0,
            kl_penalty=self.config.kl_penalty,
            target_kl=self.config.target_kl,
            init_kl_coef=self.config.init_kl_coef,
            cliprange=self.config.clip_range,
            vf_coef=self.config.vf_coef,
            log_with=None,  # manual MLflow logging
        )
        return PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    # ------------------------------------------------------------------
    def train(self, prompts: Optional[List[str]] = None) -> None:
        """Full RLHF-PPO training loop."""
        if prompts is None:
            prompts = load_prompts_from_hf(self.config.dataset_name)

        dataset = build_ppo_dataset(self.tokenizer, prompts)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name="ppo-rlhf"):
            mlflow.log_params({
                "base_model": self.config.base_model,
                "reward_model": self.config.reward_model,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "ppo_epochs": self.config.ppo_epochs,
                "kl_penalty": self.config.kl_penalty,
                "init_kl_coef": self.config.init_kl_coef,
                "clip_range": self.config.clip_range,
            })

            if self._trl_available:
                self._train_trl(dataloader)
            else:
                self._train_fallback(dataloader)

            save_path = os.path.join(self.config.output_dir, "rlhf_model")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            mlflow.log_artifacts(save_path, artifact_path="model")
            logger.info("Model saved to %s", save_path)

    def _train_trl(self, dataloader) -> None:
        ppo_trainer = self._build_ppo_trainer()
        step = 0

        for epoch in range(self.config.ppo_epochs):
            for batch in dataloader:
                if step >= self.config.max_steps:
                    return

                queries = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in batch["input_ids"]
                ]
                query_tensors = list(batch["input_ids"])

                # Generate responses from the current policy
                response_tensors = ppo_trainer.generate(query_tensors, **self.gen_kwargs)
                responses = [
                    self.tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_tensors
                ]

                # Score with reward model
                rewards = [
                    torch.tensor(s, dtype=torch.float32)
                    for s in self.reward_model.batch_score(queries, responses)
                ]

                # PPO update step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                mean_reward = torch.stack(rewards).mean().item()
                logger.info(
                    "Step %d | reward=%.4f | kl=%.4f",
                    step, mean_reward, stats.get("objective/kl", 0),
                )
                mlflow.log_metrics(
                    {
                        "mean_reward": mean_reward,
                        "kl_divergence": stats.get("objective/kl", 0),
                        "policy_loss": stats.get("ppo/loss/policy", 0),
                        "value_loss": stats.get("ppo/loss/value", 0),
                    },
                    step=step,
                )
                step += 1

    def _train_fallback(self, dataloader) -> None:
        """Minimal REINFORCE-style loop for CPU/CI environments without TRL."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        step = 0

        for batch in dataloader:
            if step >= self.config.max_steps:
                break

            queries = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["input_ids"]
            ]

            # Generate
            enc = self.tokenizer(
                queries, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **self.gen_kwargs)
            responses = self.tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Reward + KL penalty
            rewards = torch.tensor(
                self.reward_model.batch_score(queries, responses), dtype=torch.float32
            )
            with torch.no_grad():
                ref_logits = self.ref_model(**enc).logits
            policy_logits = self.model(**enc).logits
            ref_lp = torch.log_softmax(ref_logits, dim=-1)
            pol_lp = torch.log_softmax(policy_logits, dim=-1)
            kl = (pol_lp.exp() * (pol_lp - ref_lp)).sum(-1).mean()
            penalised_reward = rewards.mean() - self.config.init_kl_coef * kl

            log_probs = pol_lp.mean()
            loss = -penalised_reward * log_probs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            logger.info(
                "Step %d | reward=%.4f | kl=%.4f | loss=%.4f",
                step, rewards.mean().item(), kl.item(), loss.item(),
            )
            mlflow.log_metrics(
                {
                    "mean_reward": rewards.mean().item(),
                    "kl_divergence": kl.item(),
                    "policy_loss": loss.item(),
                    "penalised_reward": penalised_reward.item(),
                },
                step=step,
            )
            step += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sample_prompts = [
        "Summarize the key findings of this research paper:",
        "Explain the concept of attention in transformers:",
        "What are the trade-offs between RAG and fine-tuning?",
        "Describe how RLHF improves LLM alignment:",
        "Write a concise technical explanation of FAISS:",
    ] * 20

    cfg = RLHFConfig(
        base_model=os.getenv("POLICY_MODEL", "Qwen/Qwen2-0.5B-Instruct"),
        reward_model=os.getenv("REWARD_MODEL", "OpenAssistant/reward-model-deberta-v3-large-v2"),
        max_steps=int(os.getenv("MAX_STEPS", "50")),
    )
    trainer = RLHFTrainer(cfg)
    trainer.train(sample_prompts)


if __name__ == "__main__":
    main()
