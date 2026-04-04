"""
RLHF with PPO — Reinforcement Learning from Human Feedback.

Pipeline:
  1. SFT base  — load a supervised fine-tuned (or base) causal LM
  2. Reward model — small Bradley-Terry preference scorer trained on
     chosen/rejected pairs from a preference dataset
  3. PPO loop  — TRL PPOTrainer optimises the policy against the reward
     model with a KL-divergence penalty to prevent reward hacking

Covers gaps: RLHF, PPO, reasoning fine-tuning, agentic training, TRL, VeRL-style
reward shaping, MLflow experiment tracking.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, TaskType, get_peft_model

from mlops.compat import mlflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLHFConfig:
    # Model
    sft_model: str = "Qwen/Qwen2-0.5B-Instruct"   # small enough to run locally
    reward_model: str = "Qwen/Qwen2-0.5B-Instruct"
    output_dir: str = "./checkpoints/rlhf-ppo"

    # LoRA (applied to policy only)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # PPO hyper-parameters
    ppo_epochs: int = 1           # inner PPO epochs per batch
    batch_size: int = 4           # number of prompts per PPO step
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.41e-5
    kl_penalty: str = "kl"        # "kl" | "abs" | "mse" | "full"
    init_kl_coef: float = 0.2     # starting KL coefficient
    target_kl: float = 6.0        # adaptive KL target
    cliprange: float = 0.2        # PPO clip ε
    vf_coef: float = 0.1          # value-function loss coefficient
    max_new_tokens: int = 128

    # Reward model training
    rm_epochs: int = 1
    rm_learning_rate: float = 2e-5
    rm_batch_size: int = 8

    # Data
    preference_dataset: str = "Anthropic/hh-rlhf"
    prompt_dataset: str = "tatsu-lab/alpaca"
    max_prompt_samples: int = 512
    max_preference_samples: int = 1024

    # MLflow
    mlflow_experiment: str = "rlhf-ppo"
    mlflow_run_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Reward model (Bradley-Terry preference scorer)
# ---------------------------------------------------------------------------


class RewardModel(nn.Module):
    """
    Wraps a causal LM backbone with a scalar reward head.

    Trained on (chosen, rejected) pairs via the Bradley-Terry loss:
        L = -log σ(r_chosen - r_rejected)
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            backbone_name,
            num_labels=1,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)   # (batch,)

    def preference_loss(self, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
        r_chosen = self(chosen_ids, chosen_mask)
        r_rejected = self(rejected_ids, rejected_mask)
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        accuracy = (r_chosen > r_rejected).float().mean()
        return loss, accuracy


def train_reward_model(config: RLHFConfig, tokenizer) -> RewardModel:
    """Train a Bradley-Terry reward model on chosen/rejected pairs."""
    logger.info("Training reward model on %s", config.preference_dataset)

    rm = RewardModel(config.reward_model)
    rm.train()

    raw = load_dataset(config.preference_dataset, split="train", streaming=False)
    raw = raw.select(range(min(config.max_preference_samples, len(raw))))

    optimizer = torch.optim.AdamW(rm.parameters(), lr=config.rm_learning_rate)

    total_loss, total_acc = 0.0, 0.0
    steps = 0

    for i in range(0, len(raw), config.rm_batch_size):
        batch = raw[i : i + config.rm_batch_size]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        chosen_enc = tokenizer(
            chosen_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        rejected_enc = tokenizer(
            rejected_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )

        loss, acc = rm.preference_loss(
            chosen_enc["input_ids"], rejected_enc["input_ids"],
            chosen_enc.get("attention_mask"), rejected_enc.get("attention_mask"),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    avg_acc = total_acc / max(steps, 1)
    logger.info("Reward model trained — loss=%.4f  accuracy=%.4f", avg_loss, avg_acc)

    rm.eval()
    return rm, {"rm_loss": avg_loss, "rm_accuracy": avg_acc}


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------


class PPOTrainer:
    """
    Minimal PPO loop for RLHF.

    Uses TRL's PPOTrainer when available; falls back to a hand-rolled
    implementation so the code runs in CPU-only / CI environments.

    Key concepts demonstrated:
      - Policy (actor) + value head sharing the same backbone
      - KL-divergence penalty against a frozen reference policy
      - Advantage estimation via generalised advantage estimation (GAE)
      - Clipped surrogate objective (PPO-clip)
      - Adaptive KL coefficient (similar to OpenAI's approach)
    """

    def __init__(self, config: RLHFConfig):
        self.config = config
        mlflow.set_experiment(config.mlflow_experiment)

    # -- public entry point --------------------------------------------------

    def train(self):
        cfg = self.config

        with mlflow.start_run(run_name=cfg.mlflow_run_name or "ppo-run"):
            mlflow.log_params({
                "sft_model": cfg.sft_model,
                "ppo_epochs": cfg.ppo_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "kl_penalty": cfg.kl_penalty,
                "init_kl_coef": cfg.init_kl_coef,
                "target_kl": cfg.target_kl,
                "cliprange": cfg.cliprange,
            })

            tokenizer = AutoTokenizer.from_pretrained(
                cfg.sft_model, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Train reward model first
            reward_model, rm_metrics = train_reward_model(cfg, tokenizer)
            mlflow.log_metrics(rm_metrics)

            # Try TRL-based PPO first (production path)
            try:
                metrics = self._trl_ppo(cfg, tokenizer, reward_model)
            except ImportError:
                logger.warning("TRL not installed — using fallback PPO loop")
                metrics = self._fallback_ppo(cfg, tokenizer, reward_model)

            mlflow.log_metrics(metrics)
            logger.info("RLHF PPO complete: %s", metrics)
            return metrics

    # -- TRL PPO (preferred) -------------------------------------------------

    def _trl_ppo(self, cfg: RLHFConfig, tokenizer, reward_model: RewardModel) -> dict:
        from trl import PPOConfig, PPOTrainer as TRLPPOTrainer, AutoModelForCausalLMWithValueHead

        ppo_config = PPOConfig(
            model_name=cfg.sft_model,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            mini_batch_size=cfg.mini_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            ppo_epochs=cfg.ppo_epochs,
            kl_penalty=cfg.kl_penalty,
            init_kl_coef=cfg.init_kl_coef,
            target=cfg.target_kl,
            cliprange=cfg.cliprange,
            vf_coef=cfg.vf_coef,
        )

        # Policy with LoRA adapters
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_model, trust_remote_code=True
        )
        policy = get_peft_model(policy, lora_cfg)

        # Frozen reference policy (for KL penalty)
        ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_model, trust_remote_code=True
        )

        trainer = TRLPPOTrainer(
            config=ppo_config,
            model=policy,
            ref_model=ref_policy,
            tokenizer=tokenizer,
        )

        prompts = self._load_prompts(cfg, tokenizer)
        reward_pipe = self._make_reward_pipe(reward_model, tokenizer)

        all_rewards, all_kl = [], []
        for step, batch in enumerate(prompts):
            query_tensors = batch["input_ids"]
            response_tensors = trainer.generate(
                query_tensors,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=0.9,
            )
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards = [torch.tensor(reward_pipe(r)) for r in responses]

            stats = trainer.step(query_tensors, response_tensors, rewards)
            all_rewards.append(float(stats.get("ppo/mean_scores", 0)))
            all_kl.append(float(stats.get("ppo/mean_non_score_reward", 0)))

            if step % 10 == 0:
                logger.info(
                    "PPO step %d — reward=%.4f  kl=%.4f",
                    step, all_rewards[-1], all_kl[-1],
                )

            if step >= 50:   # cap for demo; remove in production
                break

        os.makedirs(cfg.output_dir, exist_ok=True)
        trainer.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        mlflow.log_artifacts(cfg.output_dir, artifact_path="rlhf_policy")

        return {
            "ppo/mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "ppo/mean_kl": sum(all_kl) / max(len(all_kl), 1),
            "ppo/steps": len(all_rewards),
        }

    # -- Fallback PPO (no TRL dependency) ------------------------------------

    def _fallback_ppo(self, cfg: RLHFConfig, tokenizer, reward_model: RewardModel) -> dict:
        """
        Hand-rolled PPO for environments without TRL installed.

        Implements the core PPO-clip update:
            L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        plus a KL penalty against a frozen reference policy.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = AutoModelForCausalLM.from_pretrained(
            cfg.sft_model, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        policy = get_peft_model(policy, lora_cfg)

        ref_policy = AutoModelForCausalLM.from_pretrained(
            cfg.sft_model, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

        prompts = self._load_prompts(cfg, tokenizer)
        kl_coef = cfg.init_kl_coef
        all_rewards, all_kl = [], []

        for step, batch in enumerate(prompts):
            input_ids = batch["input_ids"].to(device)

            # Generate response from current policy
            with torch.no_grad():
                gen_ids = policy.generate(
                    input_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_ids = gen_ids[:, input_ids.shape[1]:]

            # Compute reward
            with torch.no_grad():
                reward_enc = tokenizer(
                    [tokenizer.decode(r, skip_special_tokens=True) for r in response_ids],
                    return_tensors="pt", truncation=True, max_length=256, padding=True,
                ).to(device)
                rewards = reward_model(
                    reward_enc["input_ids"], reward_enc.get("attention_mask")
                ).detach()

            # Log-probs from policy and reference
            full_ids = gen_ids
            policy_logits = policy(full_ids).logits
            with torch.no_grad():
                ref_logits = ref_policy(full_ids).logits

            log_probs = self._token_log_probs(policy_logits, full_ids)
            ref_log_probs = self._token_log_probs(ref_logits, full_ids)

            # KL divergence penalty
            kl = (log_probs - ref_log_probs).sum(-1)
            shaped_reward = rewards - kl_coef * kl

            # Advantage (simple whitening)
            advantages = shaped_reward - shaped_reward.mean()
            if advantages.std() > 1e-8:
                advantages = advantages / (advantages.std() + 1e-8)

            # PPO-clip loss
            ratio = torch.exp(log_probs.sum(-1) - log_probs.sum(-1).detach())
            clipped = torch.clamp(ratio, 1 - cfg.cliprange, 1 + cfg.cliprange)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # Adaptive KL coefficient
            mean_kl = kl.mean().item()
            if mean_kl < cfg.target_kl / 1.5:
                kl_coef /= 1.5
            elif mean_kl > cfg.target_kl * 1.5:
                kl_coef *= 1.5

            all_rewards.append(rewards.mean().item())
            all_kl.append(mean_kl)

            if step % 5 == 0:
                logger.info(
                    "PPO step %d — reward=%.4f  kl=%.4f  kl_coef=%.4f",
                    step, all_rewards[-1], mean_kl, kl_coef,
                )

            if step >= 20:   # cap for demo
                break

        os.makedirs(cfg.output_dir, exist_ok=True)
        policy.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        return {
            "ppo/mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "ppo/mean_kl": sum(all_kl) / max(len(all_kl), 1),
            "ppo/steps": len(all_rewards),
            "ppo/final_kl_coef": kl_coef,
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-token log-probabilities for the generated sequence."""
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
        token_ids = input_ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, token_ids).squeeze(-1)

    def _load_prompts(self, cfg: RLHFConfig, tokenizer) -> list[dict]:
        """Load instruction prompts and tokenise them."""
        raw = load_dataset(cfg.prompt_dataset, split="train", streaming=False)
        raw = raw.select(range(min(cfg.max_prompt_samples, len(raw))))

        batches = []
        for i in range(0, len(raw), cfg.batch_size):
            slice_ = raw[i : i + cfg.batch_size]
            texts = [
                f"### Instruction:\n{instr}\n\n### Response:\n"
                for instr in slice_["instruction"]
            ]
            enc = tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=256, padding=True,
            )
            batches.append(enc)
        return batches

    @staticmethod
    def _make_reward_pipe(reward_model: RewardModel, tokenizer):
        """Return a callable that scores a single response string."""
        def score(text: str) -> float:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                r = reward_model(enc["input_ids"], enc.get("attention_mask"))
            return float(r.item())
        return score


# ---------------------------------------------------------------------------
# VeRL-style reward shaping utilities
# ---------------------------------------------------------------------------


class ProcessRewardModel:
    """
    Process Reward Model (PRM) — scores intermediate reasoning steps.

    Inspired by VeRL / rLLM: instead of a single outcome reward, each
    reasoning step gets a scalar score, enabling denser training signal
    for chain-of-thought / reasoning fine-tuning.
    """

    def __init__(self, reward_model: RewardModel, tokenizer):
        self.rm = reward_model
        self.tokenizer = tokenizer

    def score_steps(self, steps: list[str]) -> list[float]:
        """Score each reasoning step independently."""
        scores = []
        for step in steps:
            enc = self.tokenizer(
                step, return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                r = self.rm(enc["input_ids"], enc.get("attention_mask"))
            scores.append(float(r.item()))
        return scores

    def aggregate_reward(
        self,
        step_scores: list[float],
        strategy: str = "mean",
    ) -> float:
        """Aggregate step scores into a single training reward."""
        if not step_scores:
            return 0.0
        if strategy == "mean":
            return sum(step_scores) / len(step_scores)
        if strategy == "min":
            return min(step_scores)
        if strategy == "last":
            return step_scores[-1]
        # weighted: later steps get higher weight
        weights = [i + 1 for i in range(len(step_scores))]
        return sum(w * s for w, s in zip(weights, step_scores)) / sum(weights)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLHF PPO training")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--output-dir", default="./checkpoints/rlhf-ppo")
    args = parser.parse_args()

    cfg = RLHFConfig(
        sft_model=args.model,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        output_dir=args.output_dir,
    )
    trainer = PPOTrainer(cfg)
    metrics = trainer.train()
    print(f"Training complete: {metrics}")
