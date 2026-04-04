"""
causal_inference/
-----------------
Causal analysis for retrieval quality:
  - Uplift modeling (does reranking *cause* better RAGAS scores?)
  - ATE / CATE estimation via DoWhy + EconML
  - Counterfactual retrieval evaluation
"""

from causal_inference.uplift import UpliftEstimator, UpliftConfig, UpliftResults
from causal_inference.retrieval_effect import RetrievalCausalAnalyzer, CausalEffect
from causal_inference.counterfactual import CounterfactualEvaluator, CounterfactualResult

__all__ = [
    "UpliftEstimator",
    "UpliftConfig",
    "UpliftResults",
    "RetrievalCausalAnalyzer",
    "CausalEffect",
    "CounterfactualEvaluator",
    "CounterfactualResult",
]
