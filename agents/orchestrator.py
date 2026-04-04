"""
LangGraph multi-agent orchestrator with dependency injection.

Optional post-synthesis ``consensus_gate`` runs a cross-provider truth committee
(two models + optional judge) on the same context as the synthesizer — a
quality gate before the answer is returned.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, NotRequired, Optional, TypedDict

from agents.protocols import Reranker, Retriever, Synthesizer
from mlops.compat import mlflow

if TYPE_CHECKING:
    from agents.multi_agent.consensus_orchestrator import ConsensusOrchestrator
    from agents.multi_agent.cross_provider_consensus import CrossProviderConsensusNode
    from governance.constitution import ConstitutionalClassifier

logger = logging.getLogger(__name__)

_CONSENSUS_SYSTEM = (
    "You are a precise research assistant. Answer the user's question "
    "using ONLY the provided context. Always cite your sources using [source_N] notation. "
    "If the context doesn't contain enough information, say so explicitly."
)


def _format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Same structure as ``SynthesizerAgent._build_context`` for prompt alignment."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        src = chunk.get("source", f"doc_{i + 1}")
        parts.append(f"[source_{i + 1}] (from {src}):\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    response: Dict[str, Any]
    error: str
    skip_rag: NotRequired[bool]


class Pipeline:
    """Injectable RAG pipeline -- no global singletons."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        synthesizer: Synthesizer,
        truth_committee: Optional["CrossProviderConsensusNode"] = None,
        constitutional_classifier: Optional["ConstitutionalClassifier"] = None,
        enable_attribution: bool = False,
        enable_behavioral_gate: bool = False,
        enable_policy_enforcement: bool = False,
        adversarial_orchestrator: Optional["ConsensusOrchestrator"] = None,
        enable_paragraph_provenance: bool = False,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.truth_committee = truth_committee
        self.constitutional_classifier = constitutional_classifier
        self.enable_attribution = enable_attribution
        self.enable_behavioral_gate = enable_behavioral_gate
        self.enable_policy_enforcement = enable_policy_enforcement
        self.adversarial_orchestrator = adversarial_orchestrator
        self.enable_paragraph_provenance = enable_paragraph_provenance

    # -- graph nodes (bound to self so they close over injected agents) --

    def _behavioral_gate(self, state: AgentState) -> AgentState:
        if not self.enable_behavioral_gate:
            return {**state, "skip_rag": False}
        from safety.behavioral_classifier import classify_user_prompt

        c = classify_user_prompt(state["query"])
        if c.blocked:
            return {
                **state,
                "skip_rag": True,
                "response": {
                    "answer": c.refusal_message,
                    "sources": [],
                    "tokens_used": 0,
                    "behavioral_blocked": True,
                    "behavioral_reasons": c.reasons,
                    "behavioral_safety_score": c.safety_score,
                },
            }
        return {**state, "skip_rag": False}

    def _retrieve(self, state: AgentState) -> AgentState:
        if state.get("error") or state.get("skip_rag"):
            return state
        try:
            chunks = self.retriever.retrieve(state["query"])
            return {**state, "retrieved_chunks": chunks}
        except Exception as exc:
            logger.warning("retrieve failed: %s", exc)
            return {**state, "error": str(exc)}

    def _rerank(self, state: AgentState) -> AgentState:
        if state.get("error") or state.get("skip_rag"):
            return state
        try:
            reranked = self.reranker.rerank(state["query"], state["retrieved_chunks"])
            from context_engineering.mandatory_attribution import enrich_chunks_with_attribution_ids
            from context_engineering.traceable_rag import enrich_chunks_provenance

            reranked = enrich_chunks_provenance(enrich_chunks_with_attribution_ids(reranked))
            return {**state, "reranked_chunks": reranked}
        except Exception as exc:
            logger.warning("rerank failed: %s", exc)
            return {**state, "error": str(exc)}

    def _synthesize(self, state: AgentState) -> AgentState:
        if state.get("error") or state.get("skip_rag"):
            return state
        try:
            q = state["query"]
            chunks = state["reranked_chunks"]
            feedback_suffix = ""
            response: Dict[str, Any] = {}
            max_c = 3 if self.constitutional_classifier else 1
            for attempt in range(max_c):
                response = self.synthesizer.synthesize(q + feedback_suffix, chunks)
                if not self.constitutional_classifier:
                    break
                gr = self.constitutional_classifier.grade(response.get("answer", ""))
                response["constitutional_score"] = gr.score
                response["constitutional_violations"] = gr.violations
                response["constitutional_passed"] = gr.passed
                if gr.passed:
                    break
                if attempt < max_c - 1:
                    vtxt = "; ".join(gr.violations[:5]) if gr.violations else gr.rationale
                    feedback_suffix = (
                        f"\n\n(Constitutional regeneration {attempt + 2}/3: prior scored "
                        f"{gr.score:.0f}/100. Address: {vtxt})"
                    )
                else:
                    response["constitutional_passed"] = False
            return {**state, "response": response}
        except Exception as exc:
            logger.warning("synthesize failed: %s", exc)
            return {**state, "error": str(exc)}

    def _policy_enforcement(self, state: AgentState) -> AgentState:
        if (
            state.get("error")
            or state.get("skip_rag")
            or not self.enable_policy_enforcement
        ):
            return state
        resp = state.get("response") or {}
        ans = (resp.get("answer") or "").strip()
        if not ans:
            return state
        from agents.multi_agent.policy_enforcement_agent import enforce_grounded_answer

        chunks = state.get("reranked_chunks") or []
        pe = enforce_grounded_answer(ans, chunks)
        new_resp = {
            **resp,
            "answer": pe.enforced_answer,
            "policy_dropped_sentences": pe.dropped_sentences,
            "policy_kept_sentences": pe.kept_count,
        }
        return {**state, "response": new_resp}

    def _adversarial_consensus_step(self, state: AgentState) -> AgentState:
        """
        Dual-researcher + numeric verifier + skeptic (see ``ConsensusOrchestrator``).
        Replaces ``answer`` when enabled; skips on pipeline error or empty synthesis.
        """
        if (
            self.adversarial_orchestrator is None
            or state.get("error")
            or state.get("skip_rag")
        ):
            return state
        resp = state.get("response") or {}
        if not (resp.get("answer") or "").strip():
            return state
        chunks = state.get("reranked_chunks") or []
        context_str = _format_context_chunks(chunks)
        adv_out = self.adversarial_orchestrator.run(state["query"], context_str)
        new_resp = {
            **resp,
            "answer": adv_out.final_answer,
            "adversarial_consensus": adv_out.to_dict(),
        }
        try:
            mlflow.log_param(
                "adversarial_consensus_hitl",
                "1" if adv_out.hitl_recommended else "0",
            )
            if adv_out.conflict_events:
                mlflow.log_metric("adversarial_conflict_count", float(len(adv_out.conflict_events)))
        except Exception:
            pass
        return {**state, "response": new_resp}

    def _consensus_quality_gate(self, state: AgentState) -> AgentState:
        """
        Re-query two independent providers with the same RAG context; require agreement.

        Skipped when ``truth_committee`` is None, on pipeline error, or empty synthesis.
        """
        if self.truth_committee is None or state.get("error") or state.get("skip_rag"):
            return state
        resp = state.get("response") or {}
        if not (resp.get("answer") or "").strip():
            return state

        from agents.multi_agent.cross_provider_consensus import TruthCommitteeOutcome

        chunks = state.get("reranked_chunks") or []
        context_str = _format_context_chunks(chunks)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {state['query']}"
        raw = self.truth_committee.run(user_prompt, _CONSENSUS_SYSTEM)

        new_resp = {**resp}
        halted = (
            "HALTED: Significant discrepancy detected between independent models. "
            "Human review is required."
        )

        if raw.hitl_required:
            new_resp["answer"] = halted
            new_resp["consensus_hitl"] = True
            new_resp["consensus_discrepancy"] = raw.hitl_reason
            new_resp["consensus_peer_a"] = raw.answers[0].text[:4000]
            new_resp["consensus_peer_b"] = raw.answers[1].text[:4000]
            new_resp["consensus_score"] = raw.agreement_score
        else:
            if raw.final_text.strip():
                new_resp["answer"] = raw.final_text.strip()
            new_resp["consensus_hitl"] = False
            new_resp["consensus_score"] = raw.agreement_score

        new_resp["truth_committee"] = TruthCommitteeOutcome.from_gate(
            raw, new_resp["answer"]
        ).model_dump()

        if raw.hitl_required:
            mlflow.log_param("consensus_gate", "halted")
            mlflow.log_metric("consensus_agreement", raw.agreement_score)
        else:
            mlflow.log_param("consensus_gate", "passed")
            mlflow.log_metric("consensus_agreement", raw.agreement_score)

        return {**state, "response": new_resp}

    def _attribution_step(self, state: AgentState) -> AgentState:
        if state.get("error") or not self.enable_attribution:
            return state
        resp = state.get("response") or {}
        if resp.get("behavioral_blocked"):
            return state
        ans = (resp.get("answer") or "").strip()
        if not ans:
            return state
        from agents.attribution import attribute_answer_to_chunks
        from context_engineering.mandatory_attribution import compute_grounding_confidence
        from context_engineering.traceable_rag import (
            append_paragraph_provenance,
            faithfulness_or_proxy,
            low_confidence_human_review_message,
        )
        from mlops.grounding_trace import log_grounding_metrics

        chunks = state.get("reranked_chunks") or []
        spans = attribute_answer_to_chunks(ans, chunks)
        speculative = [s for s in spans if s.get("speculative")]
        gconf = compute_grounding_confidence(ans, chunks)
        eff_faith = faithfulness_or_proxy(grounding_confidence=gconf)
        q_alert = low_confidence_human_review_message(eff_faith)
        log_grounding_metrics(
            gconf,
            n_chunks=len(chunks),
            extra={"speculative_spans": len(speculative), "effective_faithfulness": eff_faith},
        )
        new_resp = {
            **resp,
            "source_attributions": spans,
            "speculative_sentence_count": len(speculative),
            "grounding_confidence": gconf,
            "effective_faithfulness": eff_faith,
        }
        if q_alert:
            new_resp["quality_alert"] = q_alert
        if self.enable_paragraph_provenance:
            new_resp["answer_with_provenance"] = append_paragraph_provenance(ans, chunks)
        return {**state, "response": new_resp}

    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        with mlflow.start_run(nested=mlflow.active_run() is not None):
            effective_query = query
            if session_id and os.getenv("RESEARCH_LOG_INJECT", "").lower() in (
                "1",
                "true",
                "yes",
            ):
                from agents.multi_agent.research_log import ResearchLog

                block = ResearchLog().format_for_prompt(session_id)
                if block:
                    effective_query = block + "\n\nUser question:\n" + query

            mlflow.log_param("query", effective_query[:200])
            initial: AgentState = {
                "query": effective_query,
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "response": {},
                "error": "",
                "skip_rag": False,
            }
            try:
                from langgraph.graph import END, StateGraph

                graph = StateGraph(AgentState)
                graph.add_node("behavioral", self._behavioral_gate)
                graph.add_node("retrieve", self._retrieve)
                graph.add_node("rerank", self._rerank)
                graph.add_node("synthesize", self._synthesize)
                graph.add_node("policy_gate", self._policy_enforcement)
                graph.add_node("adversarial", self._adversarial_consensus_step)
                graph.add_node("consensus_gate", self._consensus_quality_gate)
                graph.add_node("attribution", self._attribution_step)
                graph.set_entry_point("behavioral")
                graph.add_conditional_edges(
                    "behavioral",
                    after_behavioral,
                    {"retrieve": "retrieve", "__end__": END},
                )
                graph.add_conditional_edges(
                    "retrieve",
                    should_continue,
                    {"rerank": "rerank", "__end__": END},
                )
                graph.add_edge("rerank", "synthesize")
                graph.add_edge("synthesize", "policy_gate")
                graph.add_edge("policy_gate", "adversarial")
                graph.add_edge("adversarial", "consensus_gate")
                graph.add_edge("consensus_gate", "attribution")
                graph.add_edge("attribution", END)
                result = graph.compile().invoke(initial)
            except Exception:
                result = self._attribution_step(
                    self._consensus_quality_gate(
                        self._adversarial_consensus_step(
                            self._policy_enforcement(
                                self._synthesize(
                                    self._rerank(
                                        self._retrieve(self._behavioral_gate(initial))
                                    )
                                )
                            )
                        )
                    )
                )

            if result.get("response"):
                mlflow.log_metric("tokens_used", result["response"].get("tokens_used", 0))
                mlflow.log_metric("sources_retrieved", len(result["reranked_chunks"]))
                if self.truth_committee is None:
                    mlflow.log_param("consensus_gate", "disabled")
            return result


def should_continue(state: AgentState) -> str:
    return "__end__" if state.get("error") else "rerank"


def after_behavioral(state: AgentState) -> str:
    return "__end__" if state.get("skip_rag") else "retrieve"


# ---------------------------------------------------------------------------
# Convenience factory + backwards-compatible wrapper
# ---------------------------------------------------------------------------

_default_pipeline: Pipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """Thread-safe lazy singleton for use in API/Celery contexts."""
    global _default_pipeline
    if _default_pipeline is None:
        with _pipeline_lock:
            if _default_pipeline is None:
                from agents.multi_agent.cross_provider_consensus import (
                    default_truth_committee_from_env,
                )
                from agents.reranker import RerankerAgent
                from agents.retriever import RetrieverAgent
                from agents.synthesizer import SynthesizerAgent

                committee = None
                if os.getenv("ENABLE_TRUTH_COMMITTEE", "").lower() in ("1", "true", "yes"):
                    committee = default_truth_committee_from_env()
                    if committee is None:
                        logger.info(
                            "ENABLE_TRUTH_COMMITTEE is set but committee is unavailable "
                            "(need OPENAI_API_KEY, ANTHROPIC_API_KEY, pip install anthropic)."
                        )

                const_classifier = None
                if os.getenv("ENABLE_CONSTITUTION_GATE", "").lower() in ("1", "true", "yes"):
                    try:
                        from governance.constitution import ConstitutionalClassifier

                        const_classifier = ConstitutionalClassifier()
                    except Exception as exc:
                        logger.warning("Constitutional gate requested but init failed: %s", exc)

                enable_attr = os.getenv("ENABLE_ATTRIBUTION", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                enable_beh = os.getenv("ENABLE_BEHAVIORAL_GATE", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                enable_pol = os.getenv("ENABLE_POLICY_ENFORCEMENT", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )

                adversarial = None
                if os.getenv("ENABLE_ADVERSARIAL_CONSENSUS", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    try:
                        from agents.multi_agent.consensus_orchestrator import (
                            ConsensusOrchestrator,
                        )

                        adversarial = ConsensusOrchestrator.from_env()
                        if adversarial is None:
                            logger.info(
                                "ENABLE_ADVERSARIAL_CONSENSUS set but orchestrator unavailable "
                                "(need OPENAI_API_KEY, ANTHROPIC_API_KEY, anthropic package)."
                            )
                    except Exception as exc:
                        logger.warning("Adversarial consensus init failed: %s", exc)

                enable_para = os.getenv("ENABLE_TRACEABLE_PARAGRAPH_TAGS", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )

                _default_pipeline = Pipeline(
                    retriever=RetrieverAgent(top_k=10),
                    reranker=RerankerAgent(top_k=5),
                    synthesizer=SynthesizerAgent(),
                    truth_committee=committee,
                    constitutional_classifier=const_classifier,
                    enable_attribution=enable_attr,
                    enable_behavioral_gate=enable_beh,
                    enable_policy_enforcement=enable_pol,
                    adversarial_orchestrator=adversarial,
                    enable_paragraph_provenance=enable_para,
                )
    return _default_pipeline


def run_pipeline(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Run RAG pipeline; optional ``session_id`` for research-log prompt injection."""
    return get_pipeline().run(query, session_id=session_id)
