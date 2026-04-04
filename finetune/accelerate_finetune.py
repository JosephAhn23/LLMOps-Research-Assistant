"""
LoRA fine-tuning with HuggingFace Accelerate.
Covers: Accelerate, mixed precision, multi-GPU, gradient accumulation
"""
import json

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/accelerate-finetuned"


def load_dataset_from_jsonl(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    return Dataset.from_list(records)


def run_accelerate_finetune(
    data_path: str = "data/domain_train.jsonl",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-4,
    lora_rank: int = 8,
    gradient_accumulation_steps: int = 4,
):
    project_config = ProjectConfiguration(project_dir=OUTPUT_DIR, logging_dir="logs/accelerate")
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with="mlflow",
    )

    accelerator.init_trackers(
        project_name="llmops-accelerate-finetune",
        config={
            "base_model": BASE_MODEL,
            "lora_rank": lora_rank,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "n_gpus": accelerator.num_processes,
            "mixed_precision": "fp16",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    raw_dataset = load_dataset_from_jsonl(data_path)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=256,
        )

    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                if outputs.loss is None:
                    raise RuntimeError(
                        "Model returned no loss. Use AutoModelForMaskedLM or "
                        "supply a custom loss function."
                    )
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().float()
            global_step += 1

            if step % 50 == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / (step + 1)
                accelerator.log({"train_loss": avg_loss, "step": global_step, "epoch": epoch})
                print(f"Epoch {epoch} | Step {step} | Loss: {avg_loss:.4f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(f"{OUTPUT_DIR}/epoch-{epoch}")
            tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch-{epoch}")

    accelerator.end_training()
    print(f"Training complete. Model saved to {OUTPUT_DIR}")
