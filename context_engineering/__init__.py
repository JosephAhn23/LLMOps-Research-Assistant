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
from context_engineering.window_manager import ContextWindowManager, WindowResult, Priority, ContextSlot
from context_engineering.chain_of_thought import (
    ChainOfThoughtBuilder,
    ChainOfThoughtConfig,
    CoTExample,
    CoTResult,
)
from context_engineering.symbol_map import (
    build_prompt_injection_block,
    generate_symbol_map_text,
    try_pyright_symbol_dump,
)
from context_engineering.context_manager import (
    ContextManager,
    ContextBudget,
    QueryRewriter,
    RetrievalCompressor,
    TokenCostOptimizer,
)
from context_engineering.mandatory_attribution import (
    build_attribution_footer,
    compute_grounding_confidence,
    enrich_chunks_with_attribution_ids,
)
from context_engineering.traceable_rag import (
    append_paragraph_provenance,
    enrich_chunks_provenance,
    faithfulness_or_proxy,
    format_chunks_for_prompt,
    low_confidence_human_review_message,
    normalize_chunk_provenance,
)

__all__ = [
    "PromptCompressor",
    "CompressionResult",
    "DynamicFewShot",
    "FewShotExample",
    "ContextWindowManager",
    "WindowResult",
    "Priority",
    "ContextSlot",
    "ChainOfThoughtBuilder",
    "ChainOfThoughtConfig",
    "CoTExample",
    "CoTResult",
    "ContextManager",
    "ContextBudget",
    "QueryRewriter",
    "RetrievalCompressor",
    "TokenCostOptimizer",
    "generate_symbol_map_text",
    "build_prompt_injection_block",
    "try_pyright_symbol_dump",
    "enrich_chunks_with_attribution_ids",
    "compute_grounding_confidence",
    "build_attribution_footer",
    "enrich_chunks_provenance",
    "normalize_chunk_provenance",
    "format_chunks_for_prompt",
    "append_paragraph_provenance",
    "faithfulness_or_proxy",
    "low_confidence_human_review_message",
]
