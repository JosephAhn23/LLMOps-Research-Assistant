def __getattr__(name):
    """Lazy imports to avoid loading mlflow/torch at import time."""
    _lazy = {
        "Pipeline": ("agents.orchestrator", "Pipeline"),
        "run_pipeline": ("agents.orchestrator", "run_pipeline"),
        "get_pipeline": ("agents.orchestrator", "get_pipeline"),
        "RetrieverAgent": ("agents.retriever", "RetrieverAgent"),
        "RetrievedChunk": ("agents.retriever", "RetrievedChunk"),
        "RerankerAgent": ("agents.reranker", "RerankerAgent"),
        "CrossEncoderReranker": ("agents.reranker", "CrossEncoderReranker"),
        "SynthesizerAgent": ("agents.synthesizer", "SynthesizerAgent"),
        "Retriever": ("agents.protocols", "Retriever"),
        "Reranker": ("agents.protocols", "Reranker"),
        "Synthesizer": ("agents.protocols", "Synthesizer"),
    }
    if name in _lazy:
        import importlib
        module_name, attr = _lazy[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "Pipeline", "run_pipeline", "get_pipeline",
    "RetrieverAgent", "RetrievedChunk",
    "RerankerAgent", "CrossEncoderReranker",
    "SynthesizerAgent",
    "Retriever", "Reranker", "Synthesizer",
]
