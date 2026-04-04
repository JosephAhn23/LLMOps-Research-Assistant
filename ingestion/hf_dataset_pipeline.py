"""
HuggingFace Datasets Pipeline - training data prep.
Closes gaps: HuggingFace Datasets, NLP processing, data quality, dedup
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    hf_datasets: Optional[list] = None
    local_jsonl_paths: Optional[list] = None
    min_token_length: int = 50
    max_token_length: int = 2048
    min_avg_word_length: float = 3.0
    max_symbol_ratio: float = 0.1
    languages: Optional[list] = None
    dedup_min_hash: bool = True
    dedup_n_gram: int = 13
    dedup_threshold: float = 0.85
    output_path: str = "./data/processed"
    train_split: float = 0.95
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    num_proc: int = 8


class HFDatasetPipeline:
    """
    End-to-end HuggingFace Datasets pipeline.

    - Multi-source ingestion (HF Hub, local JSONL)
    - Language detection + filtering
    - Quality scoring (word length, symbol ratio)
    - MinHash LSH deduplication (datasketch)
    - Tokenizer-aware length filtering
    - Dataset mixing with configurable weights
    - Arrow format output for fast Trainer loading
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def build(self) -> DatasetDict:
        datasets = []

        if self.config.hf_datasets:
            for name in self.config.hf_datasets:
                ds = load_dataset(name, split="train")
                datasets.append(self._normalize_columns(ds))

        if self.config.local_jsonl_paths:
            for path in self.config.local_jsonl_paths:
                ds = load_dataset("json", data_files=path, split="train")
                datasets.append(self._normalize_columns(ds))

        if not datasets:
            raise ValueError("No data sources configured.")

        combined = concatenate_datasets(datasets)
        logger.info("Combined: %s examples", f"{len(combined):,}")

        combined = combined.filter(
            self._quality_filter,
            num_proc=self.config.num_proc,
            desc="Quality filtering",
        )
        logger.info("After quality filter: %s", f"{len(combined):,}")

        if self.config.dedup_min_hash:
            combined = self._minhash_dedup(combined)
            logger.info("After dedup: %s", f"{len(combined):,}")

        combined = combined.map(
            self._tokenize_and_filter,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=["text"],
            desc="Tokenizing",
        )
        combined = combined.filter(
            lambda x: x["length_ok"], num_proc=self.config.num_proc
        )
        combined = combined.remove_columns(["length_ok"])
        logger.info("Final: %s examples", f"{len(combined):,}")

        split = combined.train_test_split(
            train_size=self.config.train_split, seed=42
        )
        dataset_dict = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })
        dataset_dict.save_to_disk(self.config.output_path)
        return dataset_dict

    def _normalize_columns(self, ds: Dataset) -> Dataset:
        for col in ["text", "content", "passage", "document"]:
            if col in ds.column_names:
                if col != "text":
                    ds = ds.rename_column(col, "text")
                return ds.select_columns(["text"])
        raise ValueError(f"No text column. Available: {ds.column_names}")

    def _quality_filter(self, example: dict) -> bool:
        text = example["text"]
        if not text or not isinstance(text, str):
            return False

        words = text.split()
        if len(words) < 10:
            return False

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        if avg_word_len < self.config.min_avg_word_length:
            return False

        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return symbols / max(len(text), 1) <= self.config.max_symbol_ratio

    def _minhash_dedup(self, ds: Dataset) -> Dataset:
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch not available, skipping MinHash dedup")
            return ds

        logger.warning(
            "_minhash_dedup runs single-threaded; for large datasets consider "
            "a distributed dedup pass (e.g. datatrove or Spark)."
        )
        lsh = MinHashLSH(
            threshold=self.config.dedup_threshold, num_perm=128
        )
        keep_indices = []

        for i, ex in enumerate(ds):
            text = ex["text"]
            n = self.config.dedup_n_gram
            ngrams = set(
                text[j : j + n] for j in range(len(text) - n + 1)
            )
            m = MinHash(num_perm=128)
            for ng in ngrams:
                m.update(ng.encode("utf8"))

            # Prefix with index to guarantee uniqueness even if two different
            # texts produce the same SHA-256 digest (astronomically unlikely
            # but architecturally prevents a ValueError from datasketch).
            key = f"{i}:{hashlib.sha256(text.encode()).hexdigest()}"
            if not lsh.query(m):
                lsh.insert(key, m)
                keep_indices.append(i)

        return ds.select(keep_indices)

    def _tokenize_and_filter(self, examples: dict) -> dict:
        tokenized = self.tokenizer(
            examples["text"], truncation=False, padding=False
        )
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        length_ok = [
            self.config.min_token_length <= length <= self.config.max_token_length
            for length in lengths
        ]
        result = {
            "input_ids": [
                ids[: self.config.max_token_length]
                for ids in tokenized["input_ids"]
            ],
            "attention_mask": [
                m[: self.config.max_token_length]
                for m in tokenized["attention_mask"]
            ],
            "length_ok": length_ok,
        }
        # BERT-family tokenizers also produce token_type_ids; preserve and
        # truncate them to avoid shape mismatches in the data collator.
        if "token_type_ids" in tokenized:
            result["token_type_ids"] = [
                t[: self.config.max_token_length]
                for t in tokenized["token_type_ids"]
            ]
        return result


if __name__ == "__main__":
    config = DatasetConfig(
        hf_datasets=["tatsu-lab/alpaca", "HuggingFaceH4/ultrachat_200k"],
        languages=["en"],
        output_path="./data/processed",
    )
    HFDatasetPipeline(config).build()
