"""
Distributed LoRA fine-tuning with Ray Train.
Covers: Distributed training (Ray), replaces single-node training
"""
import json

import mlflow
import ray
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from ray import train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface import TransformersTrainer
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NUM_WORKERS = 4


def get_trainer_per_worker(config: dict):
    """
    Trainer init function - called once per Ray worker.
    Each worker gets its data shard automatically via Ray Train.
    """
    from transformers import Trainer

    model_name = config["model_name"]
    lora_rank = config["lora_rank"]
    epochs = config["epochs"]
    data_path = config["data_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_config)

    with open(data_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    dataset = Dataset.from_list(records)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=train.get_context().get_trial_dir(),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        no_cuda=False,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )


def run_distributed_finetune(data_path: str = "data/domain_train.jsonl"):
    ray.init(ignore_reinit_error=True)

    config = {
        "model_name": BASE_MODEL,
        "lora_rank": 8,
        "epochs": 3,
        "data_path": data_path,
    }

    trainer = TransformersTrainer(
        trainer_init_per_worker=get_trainer_per_worker,
        trainer_init_config=config,
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 4},
        ),
        run_config=RunConfig(
            name="lora-distributed-finetune",
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="train_loss",
            ),
        ),
    )

    with mlflow.start_run(run_name="ray-distributed-lora"):
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "lora_rank": config["lora_rank"],
                "n_workers": NUM_WORKERS,
                "epochs": config["epochs"],
            }
        )

        result = trainer.fit()

        mlflow.log_metric("final_loss", result.metrics.get("train_loss", 0))
        mlflow.log_param("checkpoint_path", str(result.checkpoint))
        print(f"Training complete. Best checkpoint: {result.checkpoint}")

    ray.shutdown()
    return result


if __name__ == "__main__":
    run_distributed_finetune()
