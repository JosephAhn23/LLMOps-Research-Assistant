"""
Reasoning Fine-Tuning with GRPO (Group Relative Policy Optimization).

Covers: RL for reasoning, chain-of-thought alignment, VeRL/SkyRL-style training,
        TRL GRPOTrainer, <think>/<answer> format enforcement, MLflow tracking.

Architecture:
  - GRPOTrainer (TRL)    — samples num_generations completions per prompt,
                           scores each with reward_funcs, normalises within group,
                           no separate value network needed (vs PPO)
  - Reward functions     — composable, passed as a list to GRPOTrainer:
      reasoning_format_reward   : enforces <think>...</think><answer>...</answer>
      math_correctness_reward   : exact-match answer extraction
      combined_reward           : weighted sum (0.3 format + 0.7 correctness)
  - build_reasoning_dataset     : chat-template prompt builder
  - ReasoningRLTrainer          : thin wrapper with MLflow tracking
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List

from datasets import Dataset, load_dataset

from mlops.compat import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a reasoning assistant. "
    "Always think step by step inside <think>...</think> tags, "
    "then give your final answer inside <answer>...</answer> tags."
)


# ---------------------------------------------------------------------------
# Reward functions
# (TRL GRPOTrainer calls these with completions + any extra dataset columns)
# ---------------------------------------------------------------------------

def reasoning_format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward correct chain-of-thought formatting.
    Expects <think>...</think><answer>...</answer> structure.
    """
    rewards = []
    for c in completions:
        has_think = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", c, re.DOTALL))
        score = 0.0
        if has_think:
            score += 0.5
        if has_answer:
            score += 0.5
        rewards.append(score)
    return rewards


def math_correctness_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward exact answer match for math/reasoning tasks.
    Extracts answer from <answer> tags and compares to ground truth.
    Falls back to numeric near-match (within 1%) for float answers.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        pred = match.group(1).strip() if match else ""
        gt = gt.strip()

        if pred == gt:
            rewards.append(1.0)
            continue

        # Numeric near-match
        try:
            pred_num = float(re.sub(r"[^\d.\-]", "", pred))
            gt_num = float(re.sub(r"[^\d.\-]", "", gt))
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.01:
                rewards.append(0.9)
                continue
        except (ValueError, ZeroDivisionError):
            pass

        rewards.append(0.0)
    return rewards


def combined_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """Weighted combination: 0.3 × format + 0.7 × correctness."""
    format_scores = reasoning_format_reward(completions)
    correctness_scores = math_correctness_reward(completions, ground_truths)
    return [0.3 * f + 0.7 * c for f, c in zip(format_scores, correctness_scores)]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_reasoning_dataset(examples: List[dict]) -> Dataset:
    """
    Build a chat-template dataset for GRPO training.

    examples: list of {"question": str, "answer": str}

    Each row has:
      "prompt"       — list of chat messages (system + user)
      "ground_truth" — expected answer string (passed to reward functions)
    """
    def format_prompt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "ground_truth": ex["answer"],
        }

    return Dataset.from_list([format_prompt(e) for e in examples])


def load_gsm8k_dataset(n: int = 500) -> Dataset:
    """Load GSM8K and convert to reasoning dataset format."""
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.select(range(min(n, len(ds))))
        examples = []
        for row in ds:
            answer = row["answer"]
            # GSM8K stores answer as "...\n#### 42" — extract the numeric part
            if "####" in answer:
                answer = answer.split("####")[-1].strip()
            examples.append({"question": row["question"], "answer": answer})
        return build_reasoning_dataset(examples)
    except Exception:
        logger.warning("Could not load GSM8K — using synthetic math dataset")
        return build_reasoning_dataset([
            {"question": "What is 15% of 240?", "answer": "36"},
            {"question": "If a train travels 120km in 2 hours, what is its speed in km/h?", "answer": "60"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": "40"},
            {"question": "If x + 7 = 15, what is x?", "answer": "8"},
        ] * 20)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReasoningRLConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./outputs/reasoning_rl"
    num_train_epochs: int = 1
    max_steps: int = 100
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 4
    num_generations: int = 4        # G in GRPO — completions sampled per prompt
    max_new_tokens: int = 256
    temperature: float = 0.9
    experiment_name: str = "grpo-reasoning"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ReasoningRLTrainer:
    """
    Thin wrapper around TRL's GRPOTrainer with MLflow tracking.

    Falls back to the hand-rolled GRPO implementation when TRL is unavailable
    (CPU/CI environments).
    """

    def __init__(self, config: ReasoningRLConfig):
        self.config = config
        self._trl_available = self._check_trl()

    def _check_trl(self) -> bool:
        try:
            from trl import GRPOTrainer, GRPOConfig  # noqa: F401
            return True
        except ImportError:
            logger.warning("TRL GRPOTrainer not available — using hand-rolled fallback")
            return False

    def train(self, dataset: Dataset) -> None:
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name="grpo-reasoning"):
            mlflow.log_params({
                "base_model": self.config.base_model,
                "num_generations": self.config.num_generations,
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
                "per_device_train_batch_size": self.config.per_device_train_batch_size,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
            })

            if self._trl_available:
                self._train_trl(dataset)
            else:
                self._train_fallback(dataset)

    def _train_trl(self, dataset: Dataset) -> None:
        from trl import GRPOConfig, GRPOTrainer

        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            num_generations=self.config.num_generations,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            log_completions=True,
            report_to="none",   # MLflow handled manually above
        )

        trainer = GRPOTrainer(
            model=self.config.base_model,
            reward_funcs=[
                reasoning_format_reward,
                math_correctness_reward,
            ],
            args=grpo_config,
            train_dataset=dataset,
        )

        logger.info("Starting GRPO reasoning fine-tuning (TRL)...")
        trainer.train()
        trainer.save_model(self.config.output_dir)
        logger.info("Model saved to %s", self.config.output_dir)

    def _train_fallback(self, dataset: Dataset) -> None:
        """
        Hand-rolled GRPO for CPU/CI environments without TRL.
        Implements the same group-relative normalisation logic.
        """
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = AutoModelForCausalLM.from_pretrained(self.config.base_model).to(device)
        ref_model = AutoModelForCausalLM.from_pretrained(self.config.base_model).to(device)
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=self.config.learning_rate)
        rows = list(dataset)

        logger.info("Starting GRPO reasoning fine-tuning (fallback)...")
        for step in range(min(self.config.max_steps, len(rows))):
            row = rows[step % len(rows)]
            # Build plain-text prompt from chat messages
            messages = row["prompt"]
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            ) + "\nASSISTANT:"
            ground_truth = row["ground_truth"]

            # Sample G responses
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            responses = []
            for _ in range(self.config.num_generations):
                with torch.no_grad():
                    out = policy.generate(
                        **enc,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=self.config.temperature,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                responses.append(
                    tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                )

            # Score and normalise within group
            raw_rewards = combined_reward(
                responses, [ground_truth] * len(responses)
            )
            reward_t = torch.tensor(raw_rewards, dtype=torch.float)
            if reward_t.std() > 1e-8:
                advantages = (reward_t - reward_t.mean()) / (reward_t.std() + 1e-8)
            else:
                advantages = reward_t - reward_t.mean()

            # Policy gradient + KL penalty
            total_loss = torch.zeros(1, requires_grad=True).to(device)
            kl_sum = 0.0
            for resp, adv in zip(responses, advantages):
                full = prompt + resp
                full_enc = tokenizer(
                    full, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                prompt_len = enc["input_ids"].shape[1]
                logits = policy(**full_enc).logits
                with torch.no_grad():
                    ref_logits = ref_model(**full_enc).logits
                labels = full_enc["input_ids"][:, 1:]
                pol_lp = F.log_softmax(logits[:, :-1], dim=-1)
                ref_lp = F.log_softmax(ref_logits[:, :-1], dim=-1)
                tok_lp = pol_lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                resp_lp = tok_lp[:, prompt_len - 1:].mean()
                kl = (pol_lp - ref_lp).mean().clamp(min=0)
                kl_sum += float(kl)
                total_loss = total_loss + (-(adv * resp_lp - 0.01 * kl))

            total_loss = total_loss / self.config.num_generations
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            fmt_score = sum(reasoning_format_reward(responses)) / len(responses)
            mlflow.log_metrics({
                "reward_mean": float(reward_t.mean()),
                "reward_max": float(reward_t.max()),
                "format_reward": fmt_score,
                "kl_mean": kl_sum / self.config.num_generations,
                "loss": float(total_loss),
            }, step=step)

            if step % 10 == 0:
                logger.info(
                    "Step %d | reward=%.3f | format=%.2f | kl=%.4f | loss=%.4f",
                    step, float(reward_t.mean()), fmt_score,
                    kl_sum / self.config.num_generations, float(total_loss),
                )

        os.makedirs(self.config.output_dir, exist_ok=True)
        policy.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info("GRPO fallback training complete. Output: %s", self.config.output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    examples = [
        {"question": "What is 15% of 240?", "answer": "36"},
        {"question": "If a train travels 120km in 2 hours, what is its speed in km/h?", "answer": "60"},
        {"question": "What is the square root of 144?", "answer": "12"},
        {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": "40"},
        {"question": "If x + 7 = 15, what is x?", "answer": "8"},
    ] * 20

    dataset = build_reasoning_dataset(examples)
    cfg = ReasoningRLConfig(
        base_model=os.getenv("POLICY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        max_steps=int(os.getenv("MAX_STEPS", "20")),
        num_generations=int(os.getenv("NUM_GENERATIONS", "4")),
    )
    trainer = ReasoningRLTrainer(cfg)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
