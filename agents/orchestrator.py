"""
LangGraph multi-agent orchestrator with dependency injection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from agents.protocols import Reranker, Retriever, Synthesizer
from mlops.compat import mlflow

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict]
    reranked_chunks: List[Dict]
    response: Dict[str, Any]
    error: str


class Pipeline:
    """Injectable RAG pipeline -- no global singletons."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        synthesizer: Synthesizer,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer

    # -- graph nodes (bound to self so they close over injected agents) --

    def _retrieve(self, state: AgentState) -> AgentState:
        try:
            chunks = self.retriever.retrieve(state["query"])
            return {**state, "retrieved_chunks": chunks}
        except Exception as exc:
            logger.warning("retrieve failed: %s", exc)
            return {**state, "error": str(exc)}

    def _rerank(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            reranked = self.reranker.rerank(state["query"], state["retrieved_chunks"])
            return {**state, "reranked_chunks": reranked}
        except Exception as exc:
            logger.warning("rerank failed: %s", exc)
            return {**state, "error": str(exc)}

    def _synthesize(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            response = self.synthesizer.synthesize(state["query"], state["reranked_chunks"])
            return {**state, "response": response}
        except Exception as exc:
            logger.warning("synthesize failed: %s", exc)
            return {**state, "error": str(exc)}

    def run(self, query: str) -> Dict[str, Any]:
        with mlflow.start_run():
            mlflow.log_param("query", query[:200])
            initial: AgentState = {
                "query": query,
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "response": {},
                "error": "",
            }
            try:
                from langgraph.graph import END, StateGraph

                graph = StateGraph(AgentState)
                graph.add_node("retrieve", self._retrieve)
                graph.add_node("rerank", self._rerank)
                graph.add_node("synthesize", self._synthesize)
                graph.set_entry_point("retrieve")
                graph.add_conditional_edges(
                    "retrieve",
                    should_continue,
                    {"rerank": "rerank", "__end__": END},
                )
                graph.add_edge("rerank", "synthesize")
                graph.add_edge("synthesize", END)
                result = graph.compile().invoke(initial)
            except Exception:
                result = self._synthesize(self._rerank(self._retrieve(initial)))

            if result.get("response"):
                mlflow.log_metric("tokens_used", result["response"].get("tokens_used", 0))
                mlflow.log_metric("sources_retrieved", len(result["reranked_chunks"]))
            return result


def should_continue(state: AgentState) -> str:
    return "__end__" if state.get("error") else "rerank"


# ---------------------------------------------------------------------------
# Convenience factory + backwards-compatible wrapper
# ---------------------------------------------------------------------------

_default_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Lazy singleton for use in API/Celery contexts."""
    global _default_pipeline
    if _default_pipeline is None:
        from agents.reranker import RerankerAgent
        from agents.retriever import RetrieverAgent
        from agents.synthesizer import SynthesizerAgent

        _default_pipeline = Pipeline(
            retriever=RetrieverAgent(top_k=10),
            reranker=RerankerAgent(top_k=5),
            synthesizer=SynthesizerAgent(),
        )
    return _default_pipeline


def run_pipeline(query: str) -> Dict[str, Any]:
    """Drop-in replacement for old global-state ``run_pipeline``."""
    return get_pipeline().run(query)
