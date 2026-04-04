# LinkedIn Copy (Recruiter-Facing)

## Headline

ML Engineer | Production LLMOps/RAG Systems | Evaluation, Causal Experimentation, and Cost-Optimized Serving

## About Section

I build production-oriented LLM systems with measurable quality, latency, and cost trade-offs.  
My current project, `LLMOps Research Assistant`, is a full RAG + experimentation stack with:

- `<5 ms` FAISS retrieval and `~40-47 ms` reranking (`n=50` benchmark queries).
- RAGAS quality gate in CI (`faithfulness=0.847`, held-out baseline set, `n=8`).
- Cost modeling that shows up to `250x` reduction when switching from GPT-4o API to self-hosted quantized serving assumptions.
- Sequential experimentation (O'Brien-Fleming), CUPED, and Double ML for decision confidence beyond naive p-values.

I focus on systems that are **auditable and reproducible**: every metric includes sample size/methodology and every production claim is tied to code and artifacts.

## Featured Section Copy

| Item | Copy to paste |
|:---|:---|
| Project link | "Live demo + full code: LLMOps Research Assistant (RAG, eval CI gates, experimentation, multi-agent reliability)." |
| Results link | "Reproducible benchmark methodology with latency, RAGAS, and cost breakdowns (includes sample sizes and confidence intervals)." |
| Case studies | "Two realistic scenarios: customer support QA flow and experimentation decision workflow with SRM/CUPED/sequential testing." |

## Experience Bullet Examples

- Built a production-style RAG pipeline with FAISS retrieval, cross-encoder reranking, and LangGraph orchestration; achieved `<5 ms` retrieval and `~40 ms` reranking (`n=50`, local benchmark).
- Implemented RAGAS + MLflow evaluation tracking and CI regression gates; prevented silent quality regressions before merge.
- Designed experiment framework with O'Brien-Fleming boundaries, CUPED variance reduction, and Double ML causal estimation to reduce false-positive shipping decisions.
- Added multi-agent fault handling with circuit breakers, retries, and graceful degradation to maintain response continuity during agent failure modes.

