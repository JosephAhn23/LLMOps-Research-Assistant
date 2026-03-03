"""
PEFT LoRA Fine-tuning with HuggingFace Transformers + Accelerate.
Closes gaps: HuggingFace ecosystem, PEFT, QLoRA, distributed training, MLflow
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator

from mlops.compat import mlflow

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./checkpoints/peft-lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    mlflow_experiment: str = "peft-lora-finetune"
    mlflow_run_name: Optional[str] = None


class PEFTLoRATrainer:
    """
    Production QLoRA fine-tuning pipeline using HuggingFace PEFT.

    - 4-bit quantization via BitsAndBytes (QLoRA)
    - LoRA adapters injected into attention layers (q,k,v,o projections)
    - Accelerate multi-GPU / distributed training
    - MLflow tracking + model registry push
    - Checkpoint auto-resumption
    """

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.accelerator = Accelerator()
        mlflow.set_experiment(config.mlflow_experiment)

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    def _load_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=self._build_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    def _inject_lora(self, model):
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, cfg)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "Trainable: %s / %s (%.2f%%) — LoRA efficiency",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )
        return model

    def _prepare_dataset(self, tokenizer):
        raw = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

        def tokenize(ex):
            prompt = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex.get('input', '')}\n\n"
                f"### Response:\n{ex['output']}"
            )
            return tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        return raw.map(tokenize, batched=False, remove_columns=raw.column_names)

    def train(self):
        with mlflow.start_run(run_name=self.config.mlflow_run_name):
            mlflow.log_params({
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "load_in_4bit": self.config.load_in_4bit,
            })

            model, tokenizer = self._load_model_and_tokenizer()
            model = self._inject_lora(model)
            dataset = self._prepare_dataset(tokenizer)

            args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type=self.config.lr_scheduler_type,
                warmup_ratio=self.config.warmup_ratio,
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                report_to="mlflow",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            result = trainer.train(
                resume_from_checkpoint=self._latest_checkpoint()
            )

            mlflow.log_metrics({
                "train_loss": result.training_loss,
                "train_runtime": result.metrics["train_runtime"],
            })

            adapter_path = os.path.join(self.config.output_dir, "lora_adapter")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            mlflow.log_artifacts(adapter_path, artifact_path="lora_adapter")

            return result

    def _latest_checkpoint(self) -> Optional[str]:
        if not os.path.isdir(self.config.output_dir):
            return None
        ckpts = [
            d
            for d in os.listdir(self.config.output_dir)
            if d.startswith("checkpoint-")
        ]
        if not ckpts:
            return None
        return os.path.join(
            self.config.output_dir,
            sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1],
        )

    @staticmethod
    def load_for_inference(base_model: str, adapter_path: str):
        """Merge LoRA weights into base model — zero inference overhead."""
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        return model.merge_and_unload(), tokenizer


if __name__ == "__main__":
    PEFTLoRATrainer(FinetuneConfig()).train()
