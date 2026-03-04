"""
Research Agent: retrieves and synthesizes information.

In production: calls the full RAG pipeline (FAISS retrieval + cross-encoder
reranking + LLM synthesis). Here: implements the interface with a simulation
that can be swapped for the real pipeline by injecting a retrieval_fn.

Tool interfaces:
  - retrieve(query) -> List[Document]
  - synthesize(query, context) -> str
  - estimate_confidence(output, context) -> float
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agents.multi_agent.base_agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    BaseAgent,
    ToolRegistry,
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Retrieves relevant documents and synthesizes a grounded answer.

    Confidence estimation:
      - High (>0.85): multiple supporting chunks, low query-context gap
      - Medium (0.6-0.85): partial context coverage
      - Low (<0.6): sparse retrieval, high uncertainty
    """

    def __init__(
        self,
        name: str = "researcher",
        retrieval_fn: Optional[Callable[[str], List[Dict]]] = None,
        synthesis_fn: Optional[Callable[[str, List[Dict]], str]] = None,
        tools: Optional[ToolRegistry] = None,
        timeout_seconds: float = 20.0,
    ):
        super().__init__(name, tools, timeout_seconds)
        self._retrieve = retrieval_fn or self._default_retrieve
        self._synthesize = synthesis_fn or self._default_synthesize

        self.tools.register("retrieve", self._retrieve, {
            "name": "retrieve",
            "description": "Retrieve relevant documents from the vector store.",
            "parameters": {"query": "string"},
        })
        self.tools.register("synthesize", self._synthesize, {
            "name": "synthesize",
            "description": "Synthesize an answer from query and retrieved context.",
            "parameters": {"query": "string", "context": "list"},
        })

    def process(self, task: AgentTask) -> AgentResult:
        query = task.query
        prior_critique = task.context.get("critique", "")

        if prior_critique:
            query = f"{query}\n\n[Improve based on critique: {prior_critique}]"
            self.logger.info("Incorporating critique feedback for task %s", task.task_id)

        chunks = self.tools.call("retrieve", query=query)
        answer = self.tools.call("synthesize", query=task.query, context=chunks)
        confidence = self._estimate_confidence(answer, chunks)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=AgentStatus.SUCCEEDED,
            output=answer,
            confidence=confidence,
            reasoning=f"Retrieved {len(chunks)} chunks. Synthesized answer.",
            sources=[c.get("doc_id", f"chunk_{i}") for i, c in enumerate(chunks)],
            tool_calls=[
                {"tool": "retrieve", "query": task.query, "n_results": len(chunks)},
                {"tool": "synthesize", "output_len": len(answer)},
            ],
            metadata={"n_chunks": len(chunks), "query_len": len(task.query)},
        )

    def _estimate_confidence(self, answer: str, chunks: List[Dict]) -> float:
        if not chunks:
            return 0.3
        coverage = sum(
            1 for c in chunks
            if any(w in answer.lower() for w in c.get("text", "").lower().split()[:10])
        ) / len(chunks)
        length_score = min(len(answer.split()) / 50, 1.0)
        return round(0.6 * coverage + 0.4 * length_score, 3)

    def _default_retrieve(self, query: str) -> List[Dict]:
        """Simulation. In production: calls ingestion.pipeline or FAISS index."""
        return [
            {"doc_id": "doc_001", "text": f"RAG uses retrieval to ground LLM responses in external knowledge, enabling up-to-date answers without retraining.", "score": 0.91},
            {"doc_id": "doc_002", "text": f"Fine-tuning embeds domain knowledge into model weights. More expensive but provides deeper stylistic consistency.", "score": 0.84},
            {"doc_id": "doc_003", "text": f"Hybrid approaches combine RAG for factual grounding with fine-tuning for domain tone and format.", "score": 0.79},
        ]

    def _default_synthesize(self, query: str, context: List[Dict]) -> str:
        ctx_text = " ".join(c["text"] for c in context[:3])
        return (
            f"Based on retrieved sources: {ctx_text[:300]}... "
            f"In summary: for '{query}', the key considerations are "
            f"retrieval quality, context window utilization, and latency tradeoffs."
        )
