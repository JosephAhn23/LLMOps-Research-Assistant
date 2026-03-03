"""
Redirects to the canonical LoRA fine-tuning implementation.

The consolidated code lives in ``finetune.lora_finetune``.  This module
re-exports its public API so older import paths keep working.
"""
from finetune.lora_finetune import *  # noqa: F401, F403
from finetune.lora_finetune import main

if __name__ == "__main__":
    main()
