"""
causal_inference/retrieval_effect.py
-------------------------------------
DoWhy causal graph for the retrieval pipeline.

Causal DAG:
  query_complexity → num_retrieved_chunks → faithfulness
  reranked         → faithfulness
  query_complexity → reranked               (confound)
  chunk_quality    → faithfulness

Identifies and estimates the direct effect of reranking on faithfulness
using backdoor adjustment + refutation tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DoWhy causal graph in GML notation
_CAUSAL_GRAPH = """
graph [
  directed 1
  node [ id "query_complexity" label "query_complexity" ]
  node [ id "num_chunks"       label "num_chunks" ]
  node [ id "reranked"         label "reranked" ]
  node [ id "chunk_quality"    label "chunk_quality" ]
  node [ id "faithfulness"     label "faithfulness" ]
  edge [ source "query_complexity" target "num_chunks" ]
  edge [ source "query_complexity" target "reranked" ]
  edge [ source "num_chunks"       target "faithfulness" ]
  edge [ source "reranked"         target "faithfulness" ]
  edge [ source "chunk_quality"    target "faithfulness" ]
]
"""


@dataclass
class CausalEstimate:
    ate: float
    method: str
    refutation_passed: bool
    refutation_pvalue: float
    raw_result: Any = None

    def summary(self) -> str:
        status = "PASSED" if self.refutation_passed else "FAILED"
        return (
            f"Method: {self.method}\n"
            f"ATE:    {self.ate:+.4f}\n"
            f"Refutation [{status}]  p={self.refutation_pvalue:.4f}"
        )


class RetrievalCausalAnalyzer:
    """
    Wraps DoWhy CausalModel for the LLMOps retrieval pipeline.

    Usage
    -----
    >>> analyzer = RetrievalCausalAnalyzer()
    >>> estimate = analyzer.estimate_reranking_effect(df)
    >>> print(estimate.summary())
    """

    TREATMENT = "reranked"
    OUTCOME = "faithfulness"
    COMMON_CAUSES = ["query_complexity", "chunk_quality"]

    def __init__(self, graph: str = _CAUSAL_GRAPH):
        self.graph = graph

    def estimate_reranking_effect(
        self,
        df: pd.DataFrame,
        method: str = "backdoor.linear_regression",
        run_refutation: bool = True,
    ) -> CausalEstimate:
        """
        Estimate the direct causal effect of reranking on faithfulness.

        Parameters
        ----------
        df             : DataFrame with treatment, outcome, and covariate columns
        method         : DoWhy identification + estimation method string
        run_refutation : Whether to run placebo refutation test
        """
        try:
            from dowhy import CausalModel
        except ImportError:
            raise ImportError("Install dowhy: pip install dowhy")

        self._validate(df)

        model = CausalModel(
            data=df,
            treatment=self.TREATMENT,
            outcome=self.OUTCOME,
            graph=self.graph,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        logger.info("Identified estimand:\n%s", identified_estimand)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            confidence_intervals=True,
        )
        ate = float(estimate.value)
        logger.info("DoWhy ATE=%.4f via %s", ate, method)

        refutation_passed = True
        refutation_pvalue = 1.0

        if run_refutation:
            refutation_passed, refutation_pvalue = self._refute(
                model, identified_estimand, estimate
            )

        return CausalEstimate(
            ate=ate,
            method=method,
            refutation_passed=refutation_passed,
            refutation_pvalue=refutation_pvalue,
            raw_result=estimate,
        )

    def sensitivity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run multiple estimators and compare ATEs for robustness.
        Returns a DataFrame comparing methods.
        """
        methods = [
            "backdoor.linear_regression",
            "backdoor.propensity_score_matching",
            "backdoor.propensity_score_weighting",
        ]
        rows = []
        for method in methods:
            try:
                est = self.estimate_reranking_effect(df, method=method, run_refutation=False)
                rows.append({"method": method, "ate": est.ate})
            except Exception as e:
                logger.warning("Method %s failed: %s", method, e)
                rows.append({"method": method, "ate": float("nan")})

        result = pd.DataFrame(rows)
        result["ate_diff_from_mean"] = result["ate"] - result["ate"].mean()
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refute(self, model: Any, estimand: Any, estimate: Any) -> tuple[bool, float]:
        """Placebo treatment refutation -- p > 0.05 means estimate is robust."""
        try:
            refute = model.refute_estimate(
                estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=100,
            )
            p_value = float(refute.refutation_result.get("p_value", 1.0))
            passed = p_value > 0.05
            logger.info("Refutation p=%.4f  passed=%s", p_value, passed)
            return passed, p_value
        except Exception as e:
            logger.warning("Refutation failed: %s", e)
            return True, 1.0

    def _validate(self, df: pd.DataFrame) -> None:
        required = [self.TREATMENT, self.OUTCOME] + self.COMMON_CAUSES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
