# Resume Talking Points

- **Deployed** a production-style RAG API on public infrastructure using `FastAPI` + `uvicorn`, resulting in a recruiter-testable live endpoint linked directly from `README.md` (method: live health check + interactive `/query` calls).
- **Recorded** an end-to-end product demo using the live pipeline (`query -> FAISS retrieval -> reranking -> grounded generation`), resulting in a 20-30 second proof artifact at `assets/quick_demo.mp4` for interview screening.
- **Built** a two-stage retrieval stack using `FAISS` bi-encoder search plus cross-encoder reranking, resulting in `<5 ms` vector search and `~40-47 ms` reranking on local benchmark runs (`n=50` queries, `benchmarks/run_benchmarks.py`).
- **Engineered** a LangGraph-based orchestration flow with retrieval, reranking, and synthesis agents, resulting in grounded responses with source citations and measurable RAGAS quality baselines (`faithfulness=0.847`, `n=8` eval pairs).
- **Instrumented** response quality evaluation with `RAGAS` + `MLflow`, resulting in regression visibility across faithfulness/relevancy/precision/recall with baseline tracking in `mlops/ragas_baseline.json`.
- **Implemented** a CI quality gate using `cicd/ragas_gate.py`, resulting in automatic PR blocking when quality regressions exceed threshold (method: metric delta check against stored baseline).
- **Optimized** inference economics by comparing GPT API vs self-hosted vLLM variants, resulting in modeled `125x-250x` lower cost per query under fp16/int4 assumptions documented in `RESULTS.md`.
- **Added** context-compression controls in the retrieval pipeline, resulting in `~31-35%` token reduction at near-flat faithfulness change (method: before/after token accounting over `n=1,000` sampled queries).
- **Developed** experimentation guardrails using SRM checks, CUPED variance reduction, and O'Brien-Fleming sequential testing, resulting in earlier statistically valid decisions without inflating false positives.
- **Applied** Double ML causal estimation with cross-fitting to experiment analysis, resulting in treatment-effect estimates robust to observed confounding (`experimentation/double_ml.py`).
- **Designed** a multi-agent supervisor with circuit breakers and graceful degradation, resulting in resilient behavior when agent calls fail instead of full pipeline collapse (`agents/multi_agent/supervisor.py`).
- **Integrated** observability with OpenTelemetry spans and optional LangSmith tracing, resulting in trace-level debugging for routing decisions, retries, and per-agent latency.
- **Implemented** governance safeguards (PII redaction, audit logging, fairness checks) using modules in `governance/`, resulting in auditable and compliance-aware LLM pipeline behavior.
- **Built** streaming/drift monitoring components using Page-Hinkley and ADWIN style detectors, resulting in faster detection of distribution shift versus periodic offline checks (`streaming/` modules).
- **Documented** architecture and trade-offs with reproducibility-first artifacts (`ARCHITECTURE.md`, `RESULTS.md`, `case_studies/`), resulting in interview-ready narratives tied to concrete files and measurable outcomes.

