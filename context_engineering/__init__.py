"""
context_engineering/
---------------------
Techniques for maximising LLM performance through context construction:

  - PromptCompressor   : LLMLingua-style token pruning
  - DynamicFewShot     : retrieve semantically relevant examples at runtime
  - ContextWindowMgr   : fit context into token budget with priority eviction
  - ChainOfThought     : structured CoT prompt builder
"""

from context_engineering.compressor import PromptCompressor, CompressionResult
from context_engineering.few_shot import DynamicFewShot, FewShotExample
from context_engineering.window_manager import ContextWindowManager, WindowResult
from context_engineering.chain_of_thought import ChainOfThoughtBuilder, CoTResult
from context_engineering.context_manager import (
    ContextManager,
    ContextBudget,
    QueryRewriter,
    RetrievalCompressor,
    TokenCostOptimizer,
)

__all__ = [
    "PromptCompressor",
    "CompressionResult",
    "DynamicFewShot",
    "FewShotExample",
    "ContextWindowManager",
    "WindowResult",
    "ChainOfThoughtBuilder",
    "CoTResult",
    "ContextManager",
    "ContextBudget",
    "QueryRewriter",
    "RetrievalCompressor",
    "TokenCostOptimizer",
]
