"""Per-query cost tracking and summary statistics for LLM inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List


DEFAULT_MODEL_RATES_USD_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0050, "output": 0.0150},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
}


@dataclass
class QueryCostRecord:
    """Represents detailed cost attribution for one query."""

    model: str
    retrieval_tokens: int
    generation_input_tokens: int
    generation_output_tokens: int
    reranking_compute_usd: float
    retrieval_cost_usd: float
    generation_cost_usd: float
    total_cost_usd: float


class CostTracker:
    """Tracks cost per query and computes aggregate dashboard metrics."""

    def __init__(self, self_hosted_rate_per_1k_tokens: float = 0.00010) -> None:
        self.self_hosted_rate_per_1k_tokens = self_hosted_rate_per_1k_tokens
        self._records: List[QueryCostRecord] = []

    def log_query(
        self,
        model: str,
        retrieval_tokens: int,
        generation_input_tokens: int,
        generation_output_tokens: int,
        reranking_compute_usd: float = 0.0,
    ) -> QueryCostRecord:
        """
        Log one query and return computed cost components.

        Args:
            model: One of "gpt-4o", "gpt-4o-mini", or "self_hosted".
            retrieval_tokens: Tokens consumed by retrieval context handling.
            generation_input_tokens: Prompt tokens passed to generator.
            generation_output_tokens: Completion tokens generated.
            reranking_compute_usd: Direct compute charge for reranking.
        """
        input_rate, output_rate = self._resolve_rates(model)

        retrieval_cost = (retrieval_tokens / 1000.0) * input_rate
        generation_cost = (
            (generation_input_tokens / 1000.0) * input_rate
            + (generation_output_tokens / 1000.0) * output_rate
        )
        total_cost = retrieval_cost + generation_cost + reranking_compute_usd

        record = QueryCostRecord(
            model=model,
            retrieval_tokens=retrieval_tokens,
            generation_input_tokens=generation_input_tokens,
            generation_output_tokens=generation_output_tokens,
            reranking_compute_usd=reranking_compute_usd,
            retrieval_cost_usd=round(retrieval_cost, 8),
            generation_cost_usd=round(generation_cost, 8),
            total_cost_usd=round(total_cost, 8),
        )
        self._records.append(record)
        return record

    def summarize(self, n_qps: float) -> Dict[str, float]:
        """
        Summarize tracked costs and project monthly spend at target QPS.

        Args:
            n_qps: Target queries per second for monthly projection.
        """
        if not self._records:
            return {
                "n_queries": 0,
                "avg_cost_per_query_usd": 0.0,
                "p50_cost_per_query_usd": 0.0,
                "p99_cost_per_query_usd": 0.0,
                "projected_monthly_cost_usd": 0.0,
            }

        totals = sorted(r.total_cost_usd for r in self._records)
        p50 = self._quantile(totals, 0.50)
        p99 = self._quantile(totals, 0.99)
        avg_cost = mean(totals)
        monthly_queries = n_qps * 60 * 60 * 24 * 30
        projected_monthly_cost = avg_cost * monthly_queries

        return {
            "n_queries": float(len(totals)),
            "avg_cost_per_query_usd": round(avg_cost, 8),
            "p50_cost_per_query_usd": round(p50, 8),
            "p99_cost_per_query_usd": round(p99, 8),
            "projected_monthly_cost_usd": round(projected_monthly_cost, 2),
        }

    def records(self) -> List[QueryCostRecord]:
        """Return a shallow copy of tracked records."""
        return list(self._records)

    def _resolve_rates(self, model: str) -> tuple[float, float]:
        if model in DEFAULT_MODEL_RATES_USD_PER_1K:
            rates = DEFAULT_MODEL_RATES_USD_PER_1K[model]
            return rates["input"], rates["output"]
        if model == "self_hosted":
            rate = self.self_hosted_rate_per_1k_tokens
            return rate, rate
        raise ValueError(
            "Unsupported model. Use 'gpt-4o', 'gpt-4o-mini', or 'self_hosted'."
        )

    @staticmethod
    def _quantile(sorted_vals: List[float], q: float) -> float:
        """Compute quantile with nearest-rank behavior for dashboard reporting."""
        if not sorted_vals:
            return 0.0
        idx = int(round((len(sorted_vals) - 1) * q))
        return sorted_vals[idx]

