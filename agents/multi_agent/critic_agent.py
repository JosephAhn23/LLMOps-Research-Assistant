"""
Critic Agent: evaluates research output quality.

Dimensions evaluated:
  - Factual grounding: are claims supported by retrieved context?
  - Completeness: does the answer address all aspects of the query?
  - Logical consistency: are there contradictions?
  - Uncertainty acknowledgment: does the answer hedge appropriately?
  - Source coverage: are multiple sources consulted?

Returns structured critique with improvement suggestions that the
ResearchAgent uses in the next iteration.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agents.multi_agent.base_agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    BaseAgent,
    ToolRegistry,
)

logger = logging.getLogger(__name__)

CRITIQUE_DIMENSIONS = [
    "factual_grounding",
    "completeness",
    "logical_consistency",
    "uncertainty_acknowledgment",
    "source_coverage",
]

UNCERTAINTY_PHRASES = [
    "may", "might", "generally", "typically", "often", "can vary",
    "depends", "in some cases", "it is possible", "evidence suggests",
]

CONTRADICTION_PATTERNS = [
    (r"\bbut\b.{0,50}\bhowever\b", "possible contradiction: 'but...however'"),
    (r"\balways\b.{0,100}\bnever\b", "absolute contradiction: 'always...never'"),
]


class CriticAgent(BaseAgent):
    """
    Evaluates research output on multiple quality dimensions.

    Produces:
    1. Per-dimension scores (0-1)
    2. Overall score (weighted average)
    3. Specific improvement suggestions
    4. Revised query if answer is off-topic
    """

    DIMENSION_WEIGHTS = {
        "factual_grounding": 0.35,
        "completeness": 0.25,
        "logical_consistency": 0.20,
        "uncertainty_acknowledgment": 0.10,
        "source_coverage": 0.10,
    }

    def __init__(
        self,
        name: str = "critic",
        quality_threshold: float = 0.75,
        tools: Optional[ToolRegistry] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name, tools, timeout_seconds)
        self.quality_threshold = quality_threshold

    def process(self, task: AgentTask) -> AgentResult:
        answer = task.context.get("research_result", "")
        retrieved_context = task.context.get("retrieved_context", [])
        query = task.query

        if not answer:
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error="No research result to critique.",
                confidence=0.0,
            )

        scores = self._evaluate_all(answer, query, retrieved_context)
        overall = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in CRITIQUE_DIMENSIONS
        )
        issues = self._identify_issues(scores, answer)
        suggestions = self._generate_suggestions(issues, query)

        needs_revision = overall < self.quality_threshold
        critique_text = self._format_critique(scores, overall, issues, suggestions)

        self.logger.info(
            "Critique: task=%s score=%.3f needs_revision=%s",
            task.task_id, overall, needs_revision,
        )

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=AgentStatus.SUCCEEDED,
            output=critique_text,
            confidence=overall,
            reasoning=f"Evaluated {len(CRITIQUE_DIMENSIONS)} dimensions.",
            metadata={
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "overall_score": round(overall, 3),
                "needs_revision": needs_revision,
                "issues": issues,
                "suggestions": suggestions,
            },
        )

    def _evaluate_all(
        self,
        answer: str,
        query: str,
        context: List[Dict],
    ) -> Dict[str, float]:
        return {
            "factual_grounding": self._score_factual_grounding(answer, context),
            "completeness": self._score_completeness(answer, query),
            "logical_consistency": self._score_consistency(answer),
            "uncertainty_acknowledgment": self._score_uncertainty(answer),
            "source_coverage": self._score_source_coverage(answer, context),
        }

    def _score_factual_grounding(self, answer: str, context: List[Dict]) -> float:
        if not context:
            return 0.3
        answer_words = set(answer.lower().split())
        coverage_scores = []
        for chunk in context:
            chunk_words = set(chunk.get("text", "").lower().split())
            if not chunk_words:
                continue
            overlap = len(answer_words & chunk_words) / max(len(chunk_words), 1)
            coverage_scores.append(min(overlap * 3, 1.0))
        return round(sum(coverage_scores) / max(len(coverage_scores), 1), 3)

    def _score_completeness(self, answer: str, query: str) -> float:
        query_terms = [w for w in query.lower().split() if len(w) > 3]
        if not query_terms:
            return 0.7
        answer_lower = answer.lower()
        covered = sum(1 for t in query_terms if t in answer_lower)
        length_score = min(len(answer.split()) / 80, 1.0)
        term_score = covered / len(query_terms)
        return round(0.6 * term_score + 0.4 * length_score, 3)

    def _score_consistency(self, answer: str) -> float:
        for pattern, _ in CONTRADICTION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                return 0.6
        return 0.9

    def _score_uncertainty(self, answer: str) -> float:
        answer_lower = answer.lower()
        hits = sum(1 for p in UNCERTAINTY_PHRASES if p in answer_lower)
        return min(0.5 + hits * 0.1, 1.0)

    def _score_source_coverage(self, answer: str, context: List[Dict]) -> float:
        if not context:
            return 0.3
        return min(len(context) / 3, 1.0)

    def _identify_issues(self, scores: Dict[str, float], answer: str) -> List[str]:
        issues = []
        if scores["factual_grounding"] < 0.5:
            issues.append("Low factual grounding: answer not well-supported by retrieved context.")
        if scores["completeness"] < 0.6:
            issues.append("Incomplete: answer does not address all aspects of the query.")
        if scores["logical_consistency"] < 0.7:
            issues.append("Possible logical contradiction detected.")
        if scores["uncertainty_acknowledgment"] < 0.5:
            issues.append("Missing uncertainty language: answer states facts too absolutely.")
        if len(answer.split()) < 30:
            issues.append("Answer is too brief (< 30 words).")
        return issues

    def _generate_suggestions(self, issues: List[str], query: str) -> List[str]:
        suggestions = []
        for issue in issues:
            if "factual grounding" in issue:
                suggestions.append("Cite specific retrieved passages to support each claim.")
            elif "incomplete" in issue:
                suggestions.append(f"Ensure all aspects of '{query[:50]}' are addressed.")
            elif "contradiction" in issue:
                suggestions.append("Review for contradictory statements and resolve them.")
            elif "uncertainty" in issue:
                suggestions.append("Add hedging language: 'typically', 'may', 'evidence suggests'.")
            elif "brief" in issue:
                suggestions.append("Expand the answer with more detail and examples.")
        return suggestions

    def _format_critique(
        self,
        scores: Dict[str, float],
        overall: float,
        issues: List[str],
        suggestions: List[str],
    ) -> str:
        lines = [f"Overall score: {overall:.3f}"]
        for dim in CRITIQUE_DIMENSIONS:
            lines.append(f"  {dim}: {scores[dim]:.3f}")
        if issues:
            lines.append("Issues: " + "; ".join(issues))
        if suggestions:
            lines.append("Suggestions: " + "; ".join(suggestions))
        return "\n".join(lines)
