"""
Agent protocols - testable interfaces for pipeline components.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str) -> List[Dict[str, Any]]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]: ...
