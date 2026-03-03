"""
LoRA fine-tuning on domain-specific data.
Covers: Fine-tuning (core, not optional)
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/finetuned-embedder"


def load_domain_data(path: str) -> Dataset:
    """Load jsonl: {"text": "...", "label": 0/1}"""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    return Dataset.from_list(records)


def finetune(data_path: str, epochs: int = 3, lora_rank: int = 8, model_name: str = BASE_MODEL):
    accelerator = Accelerator(mixed_precision="fp16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_domain_data(data_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in {"input_ids", "attention_mask"}])
    tokenized.set_format(type="torch")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(tokenized, batch_size=16, shuffle=True, collate_fn=collator)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    with mlflow.start_run(run_name="lora-finetune"):
        mlflow.log_param("base_model", model_name)
        mlflow.log_param("lora_rank", lora_config.r)
        mlflow.log_param("epochs", epochs)
        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += float(loss.detach().item())
                global_step += 1

            avg_loss = epoch_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="model")

    print(f"Fine-tuned model saved to {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/domain_train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--model-name", default=BASE_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args.data_path, epochs=args.epochs, lora_rank=args.lora_rank, model_name=args.model_name)
