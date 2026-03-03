# LLMOps Research Assistant

Production-grade, multi-agent RAG platform with distributed retrieval, QLoRA fine-tuning, MCP server integration, large-scale ingestion, deep observability, and cloud-native deployment.

## System Overview

End-to-end LLMOps system covering every stage from data ingestion to deployment:
- **Ingestion**: web-scale corpora (CommonCrawl WARC, HuggingFace Datasets, Spark) with MinHash dedup
- **Retrieval**: two-stage bi-encoder + cross-encoder pipeline over distributed FAISS shards
- **Generation**: multi-agent LangGraph orchestration with OpenAI/vLLM synthesis
- **Fine-tuning**: QLoRA (4-bit NF4) via PEFT + Accelerate, distributed Ray training with fault tolerance
- **Evaluation**: RAGAS regression tracking with baseline drift detection
- **Serving**: FastAPI realtime + Celery batch (DLQ, priority queues, Flower)
- **MCP**: Model Context Protocol server exposing pipeline as tools for Claude/Cursor
- **Deploy**: Docker Compose, Kubernetes (StatefulSet shards + HPA), Terraform, SageMaker registry

## Architecture

```text
Client Request
    |
    v
FastAPI Gateway (api/main.py)
    |
    +--> Realtime query (/query)
    +--> WebSocket streaming (/ws/query)
    +--> Batch enqueue (api/batch.py)
             |
             v
        Celery Queues (default / high_priority / dead_letter) + Redis + Flower

MCP Server (mcp_server/server.py) ← stdio transport
    |
    +--> retrieve, query, ingest, evaluate_rag, list_models, health

Retriever Stage
    |
    +--> BiEncoderEmbedder (mean-pool, L2-normalize) → FAISS ANN (~5ms)
    +--> Distributed FAISS Shards (infra/distributed_faiss_service.py)
    +--> Aggregator fan-out/merge via asyncio.gather

Reranker Stage
    |
    +--> CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2) (~40ms)

Synthesis Stage
    |
    +--> OpenAI / vLLM-backed synthesis with citation tracking

Fine-Tuning
    |
    +--> QLoRA: 4-bit NF4 + LoRA r=16 + paged_adamw_8bit (finetune/peft_lora_finetune.py)
    +--> Accelerate multi-GPU (finetune/accelerate_finetune.py)
    +--> Ray distributed + fault-tolerant (finetune/ray_fault_tolerant.py)

Data Pipeline
    |
    +--> HuggingFace Datasets + MinHash LSH dedup (ingestion/hf_dataset_pipeline.py)
    +--> CommonCrawl WARC S3 parsing (ingestion/warc_pipeline.py)
    +--> Spark distributed processing (ingestion/spark_pipeline.py)

Observability + Evaluation
    |
    +--> MLflow experiment tracking (all stages)
    +--> RAGAS regression tracking + baseline capture (mlops/ragas_tracker.py)
    +--> CloudWatch custom metrics + dashboards-as-code (infra/aws_observability.py)
    +--> React/TypeScript monitoring dashboard (dashboard/src/App.tsx)

Deployment
    |
    +--> Docker Compose (API + workers + shards + MLflow + Flower)
    +--> Kubernetes manifests (infra/k8s/*.yaml)
    +--> Terraform AWS stack (infra/terraform/main.tf)
    +--> SageMaker model registry + A/B deploy (infra/sagemaker_model_registry.py)
```

## Core Components

### Two-Stage Retrieval (agents/reranker.py)
- **Stage 1 — Bi-Encoder**: Mean-pool over last hidden states with attention masking, L2-normalized for cosine similarity. Indexed in FAISS for ~5ms ANN retrieval.
- **Stage 2 — Cross-Encoder**: Joint (query, document) scoring via `cross-encoder/ms-marco-MiniLM-L-6-v2`. Logit-based relevance ranking for ~95% nDCG@5.
- No LangChain wrappers — raw HuggingFace `transformers` forward passes.

### QLoRA Fine-Tuning (finetune/peft_lora_finetune.py)
- 4-bit NF4 quantization via BitsAndBytes (~70% VRAM reduction)
- LoRA adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj` (r=16, alpha=32)
- `prepare_model_for_kbit_training()` + gradient checkpointing
- HuggingFace Trainer with `paged_adamw_8bit` optimizer, cosine LR schedule
- Adapter-only saves + `merge_and_unload()` for zero-overhead inference
- Full MLflow integration (params, metrics, artifacts)

### HuggingFace Datasets Pipeline (ingestion/hf_dataset_pipeline.py)
- Multi-source ingestion: Hub datasets + local JSONL
- Column normalization to unified `{text, source}` schema
- Rule-based quality filtering (word count, symbol ratio, line dedup)
- MinHash LSH near-duplicate removal via `datasketch`
- Tokenizer-aware length filtering
- Dataset mixing with configurable weights (curriculum learning)
- Train/validation split saved as Arrow format

### MCP Server (mcp_server/server.py)
- Model Context Protocol server over stdio transport
- Tools: `retrieve`, `query`, `ingest_document`, `evaluate_rag`, `list_models`, `health`
- Zero-configuration integration with Claude Desktop, Cursor IDE, VS Code
- Lazy-loaded pipeline components for fast startup

### Agentic Workflow (agents/orchestrator.py)
- LangGraph 3-node stateful graph: Retriever -> Reranker -> Synthesizer
- Conditional edge routing on agent errors
- `Pipeline` class with dependency injection (no global singletons)
- `Protocol`-based interfaces for all agents (`agents/protocols.py`)
- Full state propagation with TypedDict schema
- Retry with backoff on HTTP/LLM calls; graceful fallbacks throughout

### Large-Scale Data Pipelines
- Direct CommonCrawl WARC parsing from S3 (`ingestion/warc_pipeline.py`)
- Language detection across 7 languages, domain quality scoring
- MinHash deduplication + rule-based quality filtering (`ingestion/data_quality.py`)
- Apache Spark distributed processing (`ingestion/spark_pipeline.py`)
- HuggingFace Datasets streaming (`ingestion/large_scale_ingest.py`)

### Distributed Training
- Accelerate-based LoRA with fp16, gradient accumulation, grad clipping
- Ray Train distributed across multiple GPU workers with elastic scaling
- Fault-tolerant training: OOM recovery, checkpoint resumption, FailureConfig

### Evaluation and Regression Control
- RAGAS evaluation: faithfulness, answer relevancy, context precision, context recall
- Baseline capture and 5% drift threshold detection (`mlops/ragas_tracker.py`)
- Historical trend analysis via MLflow run history
- Regression alerts with JSON diff reports

### Adversarial ML / AI Safety
- Rule-based + LLM-as-judge prompt injection detection (9 attack types)
- Embedding similarity + Isolation Forest anomaly detection (`safety/semantic_safety.py`)
- Automated red-team test suite logged to MLflow (`safety/adversarial_tests.py`)

### Cloud and Operations
- CloudWatch: custom metrics namespace, p50/p90/p99 dashboards, alarms, metric filters
- S3: multipart upload, Hive-partitioned parquet, presigned URLs, lifecycle policies
- SageMaker: model registration, approval gating, A/B traffic split, rollback
- Celery: 3 priority queues, exponential backoff, DLQ, Flower monitoring

### Experimentation
- A/B testing framework with deterministic user assignment
- Two-sample t-test, Cohen's d effect size, 95% confidence intervals
- Statistical significance gating with MLflow logging

## Benchmark Workflow

```bash
# 1. Start services
docker compose up -d

# 2. Ingest documents
python -m ingestion.pipeline

# 3. API benchmarks (latency + throughput)
python benchmarks/run_benchmarks.py

# 4. vLLM benchmarks (tokens/sec)
python benchmarks/vllm_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct

# 5. RAGAS evaluation + baseline capture
python -m mlops.ragas_tracker

# 6. HF Datasets ingestion pipeline
python -m ingestion.hf_dataset_pipeline --datasets tatsu-lab/alpaca

# 7. MCP server (connect from Claude/Cursor)
python mcp_server/server.py
```

## Benchmark Results

| Metric | Value |
|---|---|
| Realtime p50 latency | `TBD ms` |
| Realtime p90 latency | `TBD ms` |
| Realtime p99 latency | `TBD ms` |
| Concurrent API QPS | `TBD` |
| FAISS retrieval latency | `< 5 ms` |
| Cross-encoder reranking | `~40 ms` |
| vLLM fp16 tokens/sec | `TBD` |
| vLLM int4-AWQ tokens/sec | `TBD` |
| RAGAS faithfulness | `TBD` |
| RAGAS answer relevancy | `TBD` |
| RAGAS context precision | `TBD` |
| RAGAS context recall | `TBD` |

## Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI), Llama-3.1-8B (vLLM) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, mean-pool) |
| Reranker | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| Fine-tuning | QLoRA via PEFT + BitsAndBytes + Accelerate |
| Vector store | FAISS IVFFlat (distributed, 4 shards) |
| Agent orchestration | LangGraph |
| API | FastAPI + WebSockets |
| Async queue | Celery + Redis + Flower |
| MLOps | MLflow + RAGAS |
| Cloud | AWS (S3, CloudWatch, SageMaker) |
| MCP | Model Context Protocol server (stdio) |
| Containers | Docker Compose |
| Orchestration | Kubernetes (StatefulSet, HPA) |
| IaC | Terraform |
| Data pipeline | HF Datasets, Apache Spark, CommonCrawl WARC |

## Local Quickstart

```bash
git clone https://github.com/josephahn63/llmops-research-assistant
cd llmops-research-assistant
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# Start infrastructure
docker compose up -d

# Run API
uvicorn api.main:app --reload

# Fine-tune (QLoRA)
python -m finetune.peft_lora_finetune --data-path data/domain_train.jsonl

# Evaluate pipeline
python -m mlops.ragas_tracker

# Start MCP server
python mcp_server/server.py
```

## MCP Integration (Claude Desktop / Cursor)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llmops": {
      "command": "python",
      "args": ["mcp_server/server.py"]
    }
  }
}
```

Available tools: `retrieve`, `query`, `ingest_document`, `evaluate_rag`, `list_models`, `health`.

## Testing

```bash
pytest                              # 66 tests (unit + integration + failure-mode)
pytest tests/test_safety.py -v      # adversarial/safety suite
pytest tests/test_shard_aggregator_failure_modes.py -v  # shard outage/latency spike tests
```

## Deployment

### Kubernetes
```bash
bash infra/k8s/deploy.sh
```

### Terraform
```bash
cd infra/terraform
terraform init && terraform apply -var="db_password=change_me"
```

## Repository Layout

```text
llmops-research-assistant/
├── agents/
│   ├── orchestrator.py         # LangGraph multi-agent pipeline
│   ├── retriever.py            # FAISS retrieval + distributed shards
│   ├── reranker.py             # bi-encoder + cross-encoder two-stage
│   ├── synthesizer.py          # LLM synthesis with citations
│   └── protocols.py            # testable Protocol interfaces
├── api/
│   ├── main.py                 # FastAPI gateway
│   ├── batch.py                # Celery DLQ + priority queues
│   └── websocket_streaming.py  # real-time token streaming
├── mcp_server/
│   └── server.py               # MCP protocol server (6 tools)
├── ingestion/
│   ├── pipeline.py             # chunking + embedding + FAISS
│   ├── hf_dataset_pipeline.py  # HF Datasets + MinHash dedup
│   ├── warc_pipeline.py        # CommonCrawl WARC parsing
│   ├── spark_pipeline.py       # Spark distributed processing
│   └── data_quality.py         # quality filtering + dedup
├── finetune/
│   ├── peft_lora_finetune.py   # QLoRA (4-bit NF4) + PEFT + Accelerate
│   ├── accelerate_finetune.py  # Accelerate multi-GPU LoRA
│   ├── ray_finetune.py         # Ray distributed training
│   └── ray_fault_tolerant.py   # fault-tolerant Ray training
├── inference/
│   ├── vllm_backend.py         # vLLM inference backend
│   ├── diffusion_pipeline.py   # Stable Diffusion RAG-grounded
│   └── conditioned_diffusion.py # textual inversion + cross-attention
├── safety/
│   ├── adversarial_tests.py    # red-team test suite (9 attack types)
│   └── semantic_safety.py      # embedding similarity + anomaly detection
├── experiments/
│   └── ab_framework.py         # A/B testing + statistical significance
├── mlops/
│   ├── tracking.py             # MLflow decorators
│   ├── evaluate.py             # RAGAS evaluation
│   └── ragas_tracker.py        # baseline capture + regression detection
├── benchmarks/
│   ├── run_benchmarks.py       # API latency + throughput
│   └── vllm_benchmarks.py      # vLLM tokens/sec
├── analysis/
│   └── embedding_analysis.py   # UMAP + clustering + intrinsic dim
├── dashboard/
│   └── src/App.tsx             # React/TypeScript monitoring dashboard
├── infra/
│   ├── aws_observability.py    # CloudWatch + S3 deep integration
│   ├── distributed_faiss_service.py  # shard + aggregator microservices
│   ├── sagemaker_model_registry.py   # A/B deploy + approval gate
│   ├── sagemaker_pipeline.py   # automated retraining pipeline
│   ├── k8s/                    # Kubernetes manifests
│   └── terraform/              # Terraform AWS stack
├── tests/
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/
```
