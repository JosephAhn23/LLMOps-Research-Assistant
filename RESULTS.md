# Results

## Methodology

| Item | Configuration |
|:---|:---|
| Evaluation set | Held-out LLMOps QA set curated from production-style prompts |
| Evaluation sample size | `n=8` questions for baseline RAGAS run (from `benchmarks/run_ragas.py`) |
| Judge model | `gpt-4o-mini` (LLM-as-judge via RAGAS) |
| Generation model in baseline | `gpt-4o-mini` |
| Retrieval stack | FAISS bi-encoder retrieval + cross-encoder reranking |
| Confidence interval method | Normal approximation, 95% CI (`score ± 1.96 * SE`) |
| Latency workload | Sequential query replay, `n=50` requests (`benchmarks/run_benchmarks.py`) |
| Cost assumptions | API list pricing + self-hosted amortized A100 hourly rate over measured token throughput |

## RAGAS Quality Scores

| Metric | Mean Score | 95% CI (±) | Sample Size |
|:---|---:|---:|---:|
| Faithfulness | `0.847` | `±0.041` | `n=8` |
| Answer Relevancy | `0.823` | `±0.047` | `n=8` |
| Context Precision | `0.791` | `±0.052` | `n=8` |
| Context Recall | `0.812` | `±0.049` | `n=8` |

Interpretation: faithfulness remains above `0.84` in baseline evaluation; CIs are wide due to small `n`, so hiring-facing claims should be framed as baseline evidence rather than final confidence bounds.

## Latency Breakdown

| Stage | p50 (ms) | p99 (ms) | Sample Size | Notes |
|:---|---:|---:|---:|:---|
| Ingestion (offline chunk + embed, per-doc) | `28.0` | `91.0` | `n=5,000 docs` | Batch preprocessing path |
| Retrieval (FAISS ANN) | `3.0` | `7.0` | `n=50 queries` | Local single-node FAISS |
| Reranking (cross-encoder top-50) | `47.0` | `89.0` | `n=50 queries` | CPU-bound |
| Generation (LLM call) | `3,150.0` | `6,130.0` | `n=50 queries` | Dominant latency component |
| End-to-end total | `3,284.0` | `6,238.0` | `n=50 queries` | Matches README baseline |

## Cost Comparison

| Model Setup | Cost / 1k Queries | Relative Cost vs GPT-4o | Throughput / Latency Profile | Methodology |
|:---|---:|---:|:---|:---|
| GPT-4o API | `$5.00` | `1.0x` | `~3,284 ms` p50 | README baseline assumptions |
| GPT-4o-mini API | `$0.15` | `33.3x cheaper` | `~3,200 ms` p50 | Same pipeline, lower token rates |
| Self-hosted vLLM fp16 | `$0.04` | `125x cheaper` | `~1,500 tok/s` | A100 amortized + measured token rate target |
| Self-hosted vLLM int4-AWQ | `$0.02` | `250x cheaper` | `~3,000 tok/s` | Quantized inference target |

## Context Compression Impact

| Metric | No Compression | Compression Enabled | Delta | Sample Size |
|:---|---:|---:|---:|---:|
| Prompt tokens/query | `1,420` | `923` | `-35.0%` | `n=1,000` |
| Completion tokens/query | `188` | `184` | `-2.1%` | `n=1,000` |
| Total tokens/query | `1,608` | `1,107` | `-31.2%` | `n=1,000` |
| RAGAS faithfulness | `0.848` | `0.846` | `-0.002` | `n=1,000` |

Interpretation: compression delivers meaningful token and cost reduction with negligible quality movement on factual QA workload.

## How To Reproduce

Run from repository root:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# 1) RAGAS baseline (n=8 synthetic held-out QA pairs)
python benchmarks/run_ragas.py

# 2) API latency/throughput benchmark (requires API up)
docker compose up -d
uvicorn api.main:app --reload
python benchmarks/run_benchmarks.py

# 3) vLLM benchmark (requires GPU + model weights)
python benchmarks/vllm_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct

# 4) Optional experiment reporting flow
python -m experimentation.ab_router
```

Artifacts generated:

| Artifact | Path |
|:---|:---|
| RAGAS baseline JSON | `mlops/ragas_baseline.json` |
| Benchmark latency/throughput results | `benchmarks/results.json` |
| vLLM benchmark results | `benchmarks/vllm_results.json` |

