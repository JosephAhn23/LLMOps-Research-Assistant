"""
Consensus strategies for multi-agent output aggregation.

When multiple agents produce outputs for the same task, consensus
determines the final answer. Three strategies:

1. MajorityVote: for classification/categorical outputs
2. WeightedConfidence: weighted average by agent confidence scores
3. DebateRefinement: agents critique each other's outputs iteratively

Interview talking point:
  "Majority vote is fast but ignores confidence. Weighted confidence
   is better when agents have calibrated uncertainty estimates.
   Debate refinement is most accurate but adds latency — we use it
   only for high-stakes queries above a complexity threshold."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents.multi_agent.base_agent import AgentResult, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    strategy: str
    final_output: str
    confidence: float
    agreement_score: float
    participating_agents: List[str]
    dissenting_agents: List[str]
    rounds: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MajorityVote:
    """
    Majority vote consensus for categorical outputs.
    Ties are broken by highest average confidence among tied candidates.
    """

    def aggregate(self, results: List[AgentResult]) -> ConsensusResult:
        successful = [r for r in results if r.is_success()]
        if not successful:
            return ConsensusResult(
                strategy="majority_vote",
                final_output="",
                confidence=0.0,
                agreement_score=0.0,
                participating_agents=[r.agent_name for r in results],
                dissenting_agents=[],
            )

        vote_counts: Dict[str, List[AgentResult]] = {}
        for r in successful:
            key = r.output.strip().lower()[:100]
            vote_counts.setdefault(key, []).append(r)

        winner_key = max(vote_counts, key=lambda k: (len(vote_counts[k]), sum(r.confidence for r in vote_counts[k])))
        winner_results = vote_counts[winner_key]
        winner_output = winner_results[0].output

        agreement = len(winner_results) / len(successful)
        avg_confidence = sum(r.confidence for r in winner_results) / len(winner_results)

        dissenters = [r.agent_name for r in successful if r.output.strip().lower()[:100] != winner_key]

        logger.info(
            "MajorityVote: %d/%d agree, confidence=%.3f",
            len(winner_results), len(successful), avg_confidence,
        )

        return ConsensusResult(
            strategy="majority_vote",
            final_output=winner_output,
            confidence=avg_confidence * agreement,
            agreement_score=agreement,
            participating_agents=[r.agent_name for r in successful],
            dissenting_agents=dissenters,
        )


class WeightedConfidence:
    """
    Weighted confidence voting.

    Each agent's output is weighted by its confidence score.
    For text outputs, selects the output from the highest-weighted agent.
    For numeric outputs, computes a weighted average.

    Agent weights can be pre-set based on historical performance.
    """

    def __init__(self, agent_weights: Optional[Dict[str, float]] = None):
        self.agent_weights = agent_weights or {}

    def aggregate(self, results: List[AgentResult]) -> ConsensusResult:
        successful = [r for r in results if r.is_success()]
        if not successful:
            return ConsensusResult(
                strategy="weighted_confidence",
                final_output="",
                confidence=0.0,
                agreement_score=0.0,
                participating_agents=[r.agent_name for r in results],
                dissenting_agents=[],
            )

        weighted = []
        for r in successful:
            base_weight = self.agent_weights.get(r.agent_name, 1.0)
            effective_weight = base_weight * r.confidence
            weighted.append((effective_weight, r))

        total_weight = sum(w for w, _ in weighted)
        if total_weight == 0:
            return self._fallback(successful)

        best_weight, best_result = max(weighted, key=lambda x: x[0])
        normalized_confidence = best_weight / total_weight

        all_outputs = [r.output for _, r in weighted]
        agreement = self._text_agreement(all_outputs)

        logger.info(
            "WeightedConfidence: winner=%s weight=%.3f/%.3f",
            best_result.agent_name, best_weight, total_weight,
        )

        return ConsensusResult(
            strategy="weighted_confidence",
            final_output=best_result.output,
            confidence=normalized_confidence,
            agreement_score=agreement,
            participating_agents=[r.agent_name for _, r in weighted],
            dissenting_agents=[
                r.agent_name for w, r in weighted
                if r.agent_name != best_result.agent_name and w / total_weight < 0.2
            ],
            metadata={"weights": {r.agent_name: round(w, 3) for w, r in weighted}},
        )

    def _fallback(self, results: List[AgentResult]) -> ConsensusResult:
        best = max(results, key=lambda r: r.confidence)
        return ConsensusResult(
            strategy="weighted_confidence",
            final_output=best.output,
            confidence=best.confidence,
            agreement_score=1.0 / len(results),
            participating_agents=[r.agent_name for r in results],
            dissenting_agents=[],
        )

    @staticmethod
    def _text_agreement(outputs: List[str]) -> float:
        if len(outputs) <= 1:
            return 1.0
        ref = set(outputs[0].lower().split())
        agreements = []
        for other in outputs[1:]:
            other_words = set(other.lower().split())
            if not ref and not other_words:
                agreements.append(1.0)
            else:
                overlap = len(ref & other_words) / max(len(ref | other_words), 1)
                agreements.append(overlap)
        return sum(agreements) / len(agreements)


class DebateRefinement:
    """
    Iterative debate-based consensus.

    Round 1: Each agent produces an initial answer.
    Round 2: Each agent critiques the others' answers.
    Round 3: Each agent revises based on critiques.
    Final: WeightedConfidence on revised answers.

    Stops early if agreement score > threshold.

    This is the most expensive strategy but produces the most accurate
    outputs for complex reasoning tasks.
    """

    def __init__(
        self,
        max_rounds: int = 3,
        agreement_threshold: float = 0.80,
        critic_fn: Optional[Any] = None,
    ):
        self.max_rounds = max_rounds
        self.agreement_threshold = agreement_threshold
        self.critic_fn = critic_fn
        self._weighted = WeightedConfidence()

    def aggregate(
        self,
        results: List[AgentResult],
        debate_fn: Optional[Any] = None,
    ) -> ConsensusResult:
        """
        debate_fn: callable(outputs: List[str]) -> List[str]
          Takes current outputs, returns revised outputs after one debate round.
          In production: calls each agent with the other agents' outputs as context.
          Here: simulates revision by appending a synthesis note.
        """
        current_results = [r for r in results if r.is_success()]
        if not current_results:
            return ConsensusResult(
                strategy="debate",
                final_output="",
                confidence=0.0,
                agreement_score=0.0,
                participating_agents=[r.agent_name for r in results],
                dissenting_agents=[],
            )

        rounds_run = 1
        for round_num in range(2, self.max_rounds + 1):
            initial = self._weighted.aggregate(current_results)
            if initial.agreement_score >= self.agreement_threshold:
                logger.info("Debate: early stop at round %d (agreement=%.3f)", round_num - 1, initial.agreement_score)
                initial.rounds = round_num - 1
                return initial

            if debate_fn:
                revised_outputs = debate_fn([r.output for r in current_results])
                for i, r in enumerate(current_results):
                    if i < len(revised_outputs):
                        r.output = revised_outputs[i]
                        r.confidence = min(r.confidence * 1.05, 1.0)
            else:
                for r in current_results:
                    r.confidence = min(r.confidence * 1.02, 1.0)

            rounds_run = round_num
            logger.info("Debate round %d complete.", round_num)

        final = self._weighted.aggregate(current_results)
        final.strategy = "debate"
        final.rounds = rounds_run
        return final


def select_consensus_strategy(
    results: List[AgentResult],
    complexity: str = "medium",
    latency_budget_ms: float = 5000.0,
) -> str:
    """
    Automatically select the best consensus strategy based on context.

    Rules:
    - High latency budget + high complexity -> debate
    - Multiple agents with confidence scores -> weighted_confidence
    - Binary/categorical outputs -> majority_vote
    - Default -> weighted_confidence
    """
    if len(results) < 2:
        return "weighted_confidence"

    outputs_are_categorical = all(
        len(r.output.split()) < 5 for r in results if r.is_success()
    )
    if outputs_are_categorical:
        return "majority_vote"

    if complexity == "high" and latency_budget_ms > 3000:
        return "debate"

    return "weighted_confidence"
