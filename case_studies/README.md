# Case Studies

These scenarios demonstrate production-style evidence for quality, latency, and cost with reproducible pipeline stages.

| Case Study | Production Question | Pipeline Evidence | Artifact |
|:---|:---|:---|:---|
| `01_product_support_qa` | "Why did my API usage spike after enabling reranking?" | FAISS retrieval -> reranking -> grounded generation -> RAGAS | [Product Support QA](01_product_support_qa/README.md) |
| `02_experiment_analysis` | "Should we ship reranker-v2 to 100% traffic?" | SRM -> CUPED -> O'Brien-Fleming -> Double ML -> markdown report | [Experiment Analysis](02_experiment_analysis/README.md) |

All metrics include sample size (`n`) and methodology context so each claim can be defended in interviews.

