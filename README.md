<div align="center">

<img src="assets/hero_banner.png" alt="LLMOps Research Assistant" width="100%">

# LLMOps Research Assistant

**A production-architecture AI platform covering the full LLMOps lifecycle: ingestion, retrieval, generation, fine-tuning, evaluation, safety, and deployment.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)

---

*Not a wrapper. Not a tutorial. A production-architecture system built from raw components — distributed vector search, quantized fine-tuning, RLHF/PPO training, multimodal generation, streaming inference, safety layers, and multi-cloud deployment. Core pipeline runs locally; cloud and GPU components are fully implemented and ready to deploy.*

</div>

<br>

## MLOps Lifecycle

This project covers every stage of the MLOps lifecycle end-to-end:

| Stage | What's Here |
|:---|:---|
| **Data Pipelines** | Chunking, embedding, FAISS indexing; CommonCrawl WARC parsing; Apache Spark distributed ingestion; Spark ML feature engineering (TF-IDF, embeddings, quality signals); Delta Lake feature store; Databricks Unity Catalog integration |
| **Experimentation** | MLflow experiment tracking on every training run; RAGAS evaluation baseline capture; A/B experiment framework with statistical significance testing; prompt variant optimization via `PromptOptimizer` |
| **Model Training** | QLoRA (4-bit) fine-tuning via PEFT + Accelerate; multi-GPU via Ray Train; RLHF with PPO (TRL); GRPO reasoning fine-tuning (DeepSeek-R1/VeRL style); GBT query intent classifier (Spark MLlib) |
| **Serving** | FastAPI gateway; vLLM backend (fp16 / int4-AWQ); token streaming over WebSocket; Celery async job queue; microservices architecture with API gateway pattern |
| **Monitoring** | RAGAS regression gate in CI (blocks merge on quality drop); CloudWatch dashboards + alarms (AWS); Azure Monitor custom metrics (Azure); MLflow metric history; Flower task monitoring |
| **Retraining** | RAGAS baseline update workflow; CI/CD pipeline automatically evaluates every PR; configurable regression tolerance triggers retraining signal |

<div align="center">

</div>

<br>

## What This Does

Most AI projects stop at "call the API and print the answer." This one doesn't.

This platform handles the **full lifecycle** — from raw web data to a deployed, monitored, self-evaluating AI system:

| Stage | What's Implemented | Status |
|---|---|---|
| **Ingest** | Chunking, embedding, FAISS indexing from local files or HuggingFace Hub. WARC and Spark pipelines for CommonCrawl at scale. **Spark ML** (TF-IDF, Word2Vec, PCA, K-Means) for corpus analysis and balanced sampling. | Core pipeline: ✅ runs locally. WARC/Spark: implemented, requires S3 + cluster. |
| **Search** | Two-stage retrieval — FAISS bi-encoder scan across distributed shards, then cross-encoder reranking. Sub-50ms locally. | ✅ Runs locally (single-node and 4-shard Docker Compose). |
| **Generate** | **LangGraph agentic pipeline** — stateful multi-agent graph (retriever → reranker → synthesizer) with tool integration, memory, and conditional routing. Token streaming over WebSocket. vLLM backend for self-hosted inference. | ✅ Runs locally with GPT-4o-mini. vLLM requires a GPU server. |
| **Fine-tune** | QLoRA (4-bit NF4) via PEFT + BitsAndBytes. Multi-GPU via Accelerate. Fault-tolerant distributed training via Ray Train. **RLHF with PPO** — Bradley-Terry reward model, KL-penalised policy optimisation, process reward model for reasoning fine-tuning. | Implemented. Requires a GPU (single or multi). Not run in CI. |
| **Evaluate** | RAGAS scoring (faithfulness, relevancy, precision, recall) with MLflow tracking and regression alerts. **Wired into GitHub Actions CI** — quality gate blocks merge on regression > 5%. | ✅ Runs locally with an OpenAI key. CI gate: ✅ runs on every PR. |
| **Multimodal** | **Stable Diffusion v1.5** RAG-grounded image generation — retrieves context, conditions diffusion on retrieved text. DPM-Solver++ scheduler (20 steps vs 50, same quality). Textual inversion training on retrieved concepts. CLIP-based multimodal retrieval (text + image query fusion). | Implemented. Requires a GPU and diffusers. |
| **Secure** | Rule-based prompt injection detection, embedding-based anomaly detection, LLM-as-judge red-team suite. | ✅ Runs locally. |
| **Deploy** | Docker Compose (local), Kubernetes manifests, Terraform for **AWS** (VPC, EC2, ALB, RDS, S3), SageMaker model registry. **Azure Container Apps** + Azure ML workspace + Bicep IaC for full Azure parity. | Docker Compose: ✅ runs locally. K8s/Terraform/ACA: implemented, not deployed to live cloud. |
| **Connect** | MCP server (stdio) exposes retrieve, ingest, evaluate, and benchmark as tools for Claude Desktop / Cursor. | ✅ MCP server implemented and tested locally. |

<br>

## How It Works

<div align="center">
<img src="assets/pipeline_architecture.png" alt="Pipeline Architecture" width="90%">
</div>

<br>

```
 Request comes in
      |
      v
 API Gateway (FastAPI)
      |
      +---> Real-time query    ---> Retrieve ---> Rerank ---> Generate ---> Stream response
      +---> WebSocket stream   ---> Same pipeline, token-by-token delivery
      +---> Batch job          ---> Celery queue with retry logic and dead-letter capture
      |
 MCP Server (stdio)
      +---> Any AI agent can call: retrieve, ingest, evaluate, list models
      |
 Behind the scenes:
      +---> FAISS distributed across 4 shards, merged via async fan-out
      +---> Cross-encoder reranking for precision (top-50 → top-5)
      +---> MLflow tracks every run, every metric, every artifact
      +---> RAGAS watches for quality regression and alerts automatically
      +---> CloudWatch dashboards + alarms (implemented; requires AWS deployment)
```

<br>

## Performance

<div align="center">
<img src="assets/metrics_dashboard.png" alt="Performance Metrics" width="90%">
</div>

<br>

<div align="center">

| Metric | Value | Notes |
|:---|:---|:---|
| **Vector search latency** | `< 5 ms` | Measured locally, single-node FAISS |
| **Reranking latency** | `~40 ms` | Measured locally, cross-encoder on CPU |
| **End-to-end p50** | `3,284 ms` | Includes GPT-4o-mini API call |
| **End-to-end p99** | `6,238 ms` | Includes GPT-4o-mini API call |
| **Throughput** | `0.9 QPS` | Single node, sequential requests |
| **vLLM fp16** | `~1,500 tok/s` | Projected for A100; not locally measured |
| **vLLM int4-AWQ** | `~3,000 tok/s` | Projected for A100; not locally measured |

</div>

> **Note:** End-to-end latency was measured locally using GPT-4o-mini for synthesis. The retrieval + reranking pipeline alone completes in under 50 ms. vLLM throughput figures are architecture targets based on published benchmarks — the vLLM backend code is implemented but requires a GPU server to run.

### Quality Scores (RAGAS)

<div align="center">

| Metric | Score |
|:---|:---|
| Faithfulness | **0.847** |
| Answer Relevancy | **0.823** |
| Context Precision | **0.791** |
| Context Recall | **0.812** |

</div>

> **Note:** RAGAS scores were measured on a small held-out evaluation set using GPT-4o-mini as both the synthesis and judge model. Results will vary with different datasets and models.

<br>

## Key Capabilities

### Intelligent Retrieval
Two-stage search that balances speed and accuracy. The first pass uses a bi-encoder to scan millions of chunks in milliseconds via FAISS. The second pass runs a cross-encoder that reads each query-document pair together, catching subtle relevance that embedding similarity misses.

### Quantized Fine-Tuning (QLoRA) + RLHF
Implemented training pipeline for an 8-billion parameter model on a single GPU. 4-bit quantization compresses the base model while LoRA adapters inject trainable parameters into attention layers — less than 1% of total weights. Multi-GPU via Accelerate and fault-tolerant distributed training via Ray Train are also implemented.

**RLHF with PPO** (`finetune/rlhf_ppo.py`) extends supervised fine-tuning with reinforcement learning from human feedback:
- **Reward model** — Bradley-Terry preference scorer trained on chosen/rejected pairs from Anthropic HH-RLHF
- **PPO policy loop** — KL-penalised proximal policy optimisation with adaptive KL coefficient
- **Process Reward Model (PRM)** — scores intermediate reasoning steps for denser training signal (VeRL-style)
- TRL integration for production runs; hand-rolled fallback for CPU/CI environments

Requires a GPU to run.

### Real-Time Streaming
WebSocket endpoint streams tokens as they're generated. Clients see the response build word-by-word with live latency and throughput stats. Supports concurrent sessions with connection management.

### Safety & Red-Teaming
Multi-layered defense against adversarial inputs. Rule-based pattern matching for known attack signatures. Embedding-based anomaly detection for novel attacks. LLM-as-judge for ambiguous cases. Full red-team test suite with 9 attack categories.

### Data Pipelines at Scale (Spark + Spark ML)
Ingestion from local files and HuggingFace Hub runs locally. CommonCrawl WARC parsing (S3-backed) and Apache Spark distributed processing are fully implemented pipelines — they require an S3 bucket and a Spark cluster to run at scale. All pipelines include language detection, quality scoring, MinHash deduplication, and tokenizer-aware filtering.

**Spark ML preprocessing** (`ingestion/spark_ml_pipeline.py`) adds MLlib-based feature engineering:
- TF-IDF extraction (HashingTF → IDF → L2 normalisation) via the Spark ML Pipeline API
- Word2Vec document embeddings (128-dim, distributed training)
- PCA dimensionality reduction for visualisation and downstream ML
- K-Means clustering for corpus analysis and balanced sampling
- Quality-stratified reservoir sampling across semantic clusters

### Self-Evaluating Pipeline with CI/CD Gate
RAGAS metrics run automatically after each deployment **and on every pull request** via GitHub Actions. The `ragas-eval` CI job:
- Runs faithfulness, answer relevancy, context precision, and context recall
- Hard-fails if any metric drops below minimum thresholds (faithfulness < 0.70, etc.)
- Detects regressions > 5% against the saved baseline and blocks the merge
- Posts a metric summary table as a PR comment
- Uploads scores as a build artifact (90-day retention)

Baseline capture, regression detection with configurable thresholds, and historical trend tracking. If quality drops, you know before it ships — not after users complain.

### MCP Integration
Exposes the pipeline as tools via the Model Context Protocol. The MCP server is implemented and runs locally — connect it to Claude Desktop or Cursor using the config below and it will call your local pipeline directly. Retrieve documents, ingest new data, and run evaluations through natural language.

### Multimodal Generation (Stable Diffusion)
RAG-grounded image generation using Stable Diffusion v1.5 (`inference/diffusion_pipeline.py`, `inference/conditioned_diffusion.py`):

- **RAG-conditioned generation** — retrieves relevant text chunks, synthesises a grounded prompt, generates images conditioned on retrieved context
- **DPM-Solver++ scheduler** — 20 inference steps vs default 50, matching quality at 2.5× speed
- **Textual inversion** — learns a `<retrieved-concept>` token by fine-tuning the CLIP text encoder on retrieved chunks; MD5-based concept caching avoids redundant training
- **Cross-attention visualisation** — hooks into UNet `attn2` layers to capture and visualise which retrieved concepts influence each image region
- **Multimodal retrieval** (`MultimodalRAGPipeline`) — fuses CLIP text and image embeddings for queries that include both text and reference images

Requires a GPU and `diffusers==0.27.2`. All code is in `inference/`.

### Agentic Multi-Agent Orchestration
The LangGraph pipeline is a full **agentic system**, not just a chain:

- **Stateful graph** (`agents/orchestrator.py`) — `StateGraph` with typed `AgentState`, conditional routing, and error recovery at each node
- **Tool-using agents** — each agent (retriever, reranker, synthesizer) implements a typed `Protocol` interface, enabling hot-swapping and independent testing
- **Memory and context management** — state is passed explicitly through the graph; each node can inspect and modify the full context window
- **MCP tool exposure** — the pipeline is exposed as callable tools via the Model Context Protocol, making it composable with any MCP-compatible agent (Claude, Cursor, etc.)
- **Fault-tolerant routing** — conditional edges detect agent failures and route to fallback paths without crashing the graph

### A/B Experimentation
Built-in experiment framework with deterministic user assignment, statistical significance testing, and effect size measurement. Compare retrieval strategies, reranking models, or generation configs with confidence.

<br>

## Built With

<div align="center">

| | |
|:---|:---|
| **Models** | GPT-4o-mini, Llama-3.1-8B, Qwen2-0.5B, Stable Diffusion v2.1, CLIP, LLaVA-1.5 |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace) |
| **Reranker** | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| **Fine-tuning** | QLoRA via PEFT + BitsAndBytes + Accelerate |
| **RLHF** | PPO + Bradley-Terry reward model + Process Reward Model (TRL / hand-rolled) |
| **Reasoning RL** | GRPO (TRL GRPOTrainer) — DeepSeek-R1 / VeRL style |
| **Multimodal** | Stable Diffusion v2.1 + DPM-Solver++ + CLIP image index + LLaVA VQA |
| **Vector Store** | FAISS IndexFlatIP (distributed, 4 shards) |
| **Orchestration** | LangGraph agentic pipeline (stateful graph, conditional routing, tool protocols) |
| **Prompt Management** | LangChain (PromptTemplate, FewShotPromptTemplate, LCEL chains, output parsers) |
| **Local Inference** | llama.cpp (GGUF, Q3–F16, CPU/Metal/CUDA) + SLM eval suite (QA, reasoning, needle-in-haystack, contamination detection, conversation dynamics) |
| **Inference Serving** | TensorRT (FP16/INT8/FP8 PTQ), ONNX Runtime, Triton Inference Server (dynamic batching + ensemble pipeline) |
| **Quantization** | PTQ (INT8/NF4), GPTQ, AWQ, QAT, NVIDIA 2:4 sparsity |
| **Config Management** | Hydra (hierarchical configs, env scoping, CLI overrides, multirun sweeps) |
| **Eval UI** | Gradio (RAG query, model A/B comparison, quant benchmark, human feedback + RLHF export, metrics dashboard) |
| **Observability** | Prometheus (request latency, TTFT, token throughput, NDCG, RAGAS gauges, GPU memory) + Grafana dashboard (auto-provisioned) + SLO alert rules + Alertmanager Slack routing |
| **torch.compile** | Eager vs compiled benchmarking (inductor/cudagraphs/onnxrt), graph break detection (torch._dynamo.explain), AoT Autograd tracing, torch.export serialisation, DynamicShapeManager |
| **TensorRT-LLM** | Engine builder (FP16/INT8-SQ/INT4-AWQ/FP8), in-flight batching, paged KV-cache, Triton config |
| **NVIDIA NIM** | OpenAI-compatible NIM backend (`local()`/`cloud()` classmethods), streaming, health check, benchmark, Docker Compose generator |
| **Inference Factory** | `InferenceBackendFactory` — unified `.generate()` interface across NIM / TRT-LLM / vLLM / llama.cpp |
| **NVIDIA NIM** | OpenAI-compatible NIM backend adapter, embedding endpoint, latency benchmarking |
| **CUDA Kernels** | Custom C++ extension: fused attention clip, top-k sampling, fused RMSNorm |
| **MoE Serving** | Mixtral/DeepSeek expert parallelism config, load balancer, DeepSpeed-Inference, Ray Serve |
| **Multi-GPU** | transformers device_map (auto/balanced/sequential), pipeline parallelism layout |
| **API** | FastAPI + WebSockets |
| **Queue** | Celery + Redis + Flower |
| **Event Bus** | Kafka (aiokafka async + kafka-python sync) + Redis Streams (aioredis) |
| **Streaming Pipeline** | Kafka + AWS Kinesis producers/consumers, DLQ routing, DynamoDB checkpointing |
| **Tracking** | MLflow + RAGAS (CI-gated regression detection) |
| **Cloud — AWS** | S3, CloudWatch, SageMaker model registry |
| **Cloud — Azure** | Container Apps, Azure ML workspace, Azure OpenAI, Key Vault, Azure Monitor |
| **IaC** | Terraform (AWS) + Terraform + Bicep (Azure) |
| **Agent Protocol** | MCP server (stdio transport) |
| **Containers** | Docker Compose (monolith) + Docker Compose (microservices) + Kubernetes |
| **Data** | HF Datasets, Apache Spark, Spark ML (MLlib), Delta Lake, Databricks Unity Catalog, CommonCrawl |
| **Dashboard** | React + TypeScript |

</div>

<br>

## Quick Start

```bash
git clone https://github.com/JosephAhn23/LLMOps-Research-Assistant
cd LLMOps-Research-Assistant
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=your_key        # Mac/Linux
set OPENAI_API_KEY=your_key           # Windows

# Start services (monolith mode)
docker compose up -d

# Launch the API
uvicorn api.main:app --reload

# Fine-tune a model (QLoRA)
python -m finetune.peft_lora_finetune

# Fine-tune with RLHF/PPO (reward model + policy optimisation)
python -m finetune.rlhf_ppo --model Qwen/Qwen2-0.5B-Instruct

# RLHF with PPO (TRL-backed, full pipeline)
python rl/rlhf_pipeline.py

# GRPO reasoning fine-tuning (DeepSeek-R1 / VeRL style)
python rl/reasoning_rl.py

# Run quality evaluation
python -m mlops.ragas_tracker

# Test CI regression gate locally
python cicd/ragas_gate.py \
  --scores mlops/ragas_scores.json \
  --baseline mlops/ragas_baseline.json

# Generate images conditioned on RAG context (requires GPU + diffusers)
python -c "from inference.diffusion_pipeline import RAGGroundedDiffusion; RAGGroundedDiffusion().load_pipeline()"

# Multimodal RAG pipeline demo (CLIP retrieval, CPU-compatible)
python multimodal/multimodal_pipeline.py

# Run Spark ML preprocessing pipeline
python ingestion/spark_ml_pipeline.py --input data/raw.parquet --output data/processed.parquet

# Run Spark ML training pipeline (GBT classifier + KMeans + Delta Lake)
python spark_ml/spark_ml_pipeline.py

# Run Databricks notebook locally (or upload to Databricks workspace)
python spark_ml/databricks_notebook.py

# Test LangChain prompt layer (smoke test, no API key needed)
python agents/langchain_prompts.py

# Generate Azure Bicep IaC template
python infra/azure_deploy.py gen-bicep --output infra/azure/main.bicep

# Deploy Azure infrastructure via Terraform
# cd infra/azure && terraform init && terraform apply

# Run TensorRT/ONNX export pipeline (ONNX path runs on CPU)
python inference/tensorrt_onnx_triton.py

# Generate Triton config only (no model download)
python inference/tensorrt_onnx_triton.py --triton-only --triton-repo triton_repo

# Local inference with llama.cpp (downloads GGUF model on first run)
python inference/llamacpp_backend.py --prompt "Explain RAG"

# Full SLM evaluation suite (QA, reasoning, needle-in-haystack, contamination)
python inference/llamacpp_backend.py --model models/llama-3.2-1b-q4_k_m.gguf --eval

# Skip long-context tests for faster eval
python inference/llamacpp_backend.py --eval --no-long-context --mlflow

# Quantization demo (PTQ dynamic + pruning, CPU-compatible)
python finetune/quantization_suite.py

# Set up Kafka topics (requires running Kafka)
python streaming/streaming_pipeline.py --backend kafka --mode setup

# Produce 10 test inference events to Kafka
python streaming/streaming_pipeline.py --backend kafka --mode produce --n-events 10

# Consume and process events (Kafka)
python streaming/streaming_pipeline.py --backend kafka --mode consume --n-events 10

# Set up Kinesis streams + DynamoDB checkpoint table (requires AWS credentials)
python streaming/streaming_pipeline.py --backend kinesis --mode setup

# Generate monitoring config files
python monitoring/prometheus_monitoring.py --generate

# Start Prometheus + Grafana observability stack
docker compose -f docker-compose.observability.yml up -d
# Grafana: http://localhost:3000  Prometheus: http://localhost:9090

# torch.compile benchmark with real HuggingFace model (CPU, no GPU required)
python compile/torch_compile.py --model prajjwal1/bert-tiny --device cpu --graph-breaks

# torch.compile benchmark (all backends, GPU)
python compile/torch_compile.py --model bert-base-uncased --device cuda \
  --backends inductor cudagraphs --modes default reduce-overhead --mlflow

# AoT export for deployment
python compile/torch_compile.py --model bert-base-uncased --export-aot

# Build TRT-LLM engine (requires A100 + TRT-LLM toolkit)
python inference/tensorrt_llm_config.py build --model-name llama-3.1-8b --model-dir ./hf_models/llama3

# Generate NIM Docker Compose and start container
python inference/trtllm_nim.py --gen-docker --model meta/llama-3.1-8b-instruct
docker compose -f docker-compose.nim.yml up -d

# Check NIM container health
python inference/trtllm_nim.py --health --port 8000

# Benchmark NIM endpoint
python inference/trtllm_nim.py --benchmark --port 8000

# Generate TRT-LLM build script
python inference/trtllm_nim.py --gen-build-script --model meta-llama/Llama-3.1-8B-Instruct

# MoE parallelism layout (CPU, no GPU required)
python inference/moe_serving.py layout --model mixtral-8x7b --n-gpus 8 --ep-size 8

# Build CUDA extension (requires CUDA toolkit)
python csrc/setup.py build_ext --inplace
python csrc/kernels_wrapper.py --device cuda

# Launch Gradio evaluation UI
python eval/gradio_eval_ui.py

# Use Hydra config overrides
python config/hydra_config.py env=production model=azure_openai

# Start microservices stack (distributed mode)
docker compose -f microservices/docker-compose.microservices.yml up

# Start MCP server (for Claude/Cursor)
python mcp_server/server.py
```

### Connect to Claude Desktop or Cursor

Add to your MCP config:

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

Then ask Claude: *"Retrieve documents about RAG architectures"* — it calls your pipeline directly.

<br>

## Run Benchmarks

```bash
# API latency + throughput (hits live endpoints)
python benchmarks/fill_readme_benchmarks.py

# Standalone RAGAS evaluation (just needs an OpenAI key)
python benchmarks/run_ragas.py

# vLLM throughput (requires GPU server)
python benchmarks/vllm_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct
```

<br>

## Testing

```bash
pytest                                # Full suite (66 tests)
pytest tests/test_safety.py -v        # Adversarial/safety tests
pytest tests/test_shard_aggregator_failure_modes.py -v  # Fault tolerance
```

<br>

## Deploy

The deployment layer is fully implemented — manifests and IaC are ready to apply against a real cluster or AWS account.

**Kubernetes** (requires a running cluster):
```bash
bash infra/k8s/deploy.sh
```

**Terraform (AWS)** (provisions VPC, EC2, ALB, RDS, S3, Route53):
```bash
cd infra/terraform
terraform init && terraform apply
```

<br>

## Project Structure

```
llmops-research-assistant/
|
|-- agents/                     # Multi-agent pipeline
|   |-- orchestrator.py            LangGraph stateful workflow
|   |-- retriever.py               FAISS search + distributed shards
|   |-- reranker.py                Bi-encoder + cross-encoder two-stage
|   |-- synthesizer.py             LLM generation with citations
|   |-- langchain_prompts.py       LangChain prompt templates, LCEL chains, output parsers
|   +-- protocols.py               Testable interfaces
|
|-- api/                        # API layer
|   |-- main.py                    FastAPI gateway
|   |-- batch.py                   Celery queues + dead-letter handling
|   +-- websocket_streaming.py     Real-time token streaming
|
|-- mcp_server/                 # Agent integration
|   +-- server.py                  MCP protocol (6 tools, stdio)
|
|-- ingestion/                  # Data pipelines
|   |-- pipeline.py                Chunking + embedding + FAISS indexing
|   |-- hf_dataset_pipeline.py     HuggingFace Datasets + MinHash dedup
|   |-- warc_pipeline.py           CommonCrawl WARC parsing
|   |-- spark_pipeline.py          Spark distributed processing
|   +-- spark_ml_pipeline.py       Spark ML — TF-IDF, Word2Vec, PCA, K-Means (MLlib)
|
|-- finetune/                   # Model training
|   |-- peft_lora_finetune.py      QLoRA (4-bit) + PEFT + Accelerate
|   |-- rlhf_ppo.py                RLHF — reward model + PPO + Process Reward Model
|   |-- quantization_suite.py      PTQ / GPTQ / AWQ / QAT / sparsity / pruning
|   |-- ray_fault_tolerant.py      Ray distributed + fault recovery
|   +-- accelerate_finetune.py     Multi-GPU LoRA
|
|-- rl/                         # Reinforcement learning fine-tuning
|   |-- rlhf_pipeline.py           PPO + frozen KL reference model + RewardModel (TRL)
|   +-- reasoning_rl.py            GRPO reasoning fine-tuning (DeepSeek-R1 / VeRL style)
|
|-- inference/                  # Model serving
|   |-- vllm_backend.py            vLLM inference backend
|   |-- llamacpp_backend.py         llama.cpp local inference (GGUF, Q3–F16, CPU/Metal/CUDA)
|   |                               + SLMEvaluator (QA/reasoning/long-context/contamination/conversation)
|   |                               + QuantizationComparison (Q4 vs Q5 vs Q8 vs F16)
|   |-- tensorrt_onnx_triton.py     ONNX export, TensorRT FP16/INT8/FP8 PTQ, Triton config + ensemble + HTTP client
|   |-- tensorrt_llm_config.py     TRT-LLM engine builder (SQ/AWQ/FP8), NIM backend adapter, latency benchmark
|   |-- torch_compile_bench.py     torch.compile eager vs compiled benchmark (inductor/aot_eager/cudagraphs)
|   |-- moe_serving.py             MoE router (top-k/sigmoid), expert parallelism, DeepSpeed-Inference, Ray Serve
|   |-- diffusion_pipeline.py      Stable Diffusion v2.1 + RAG grounding + CLIP multimodal
|   +-- conditioned_diffusion.py   Textual inversion + cross-attention visualisation
|
|-- multimodal/                 # Multimodal RAG pipeline
|   +-- multimodal_pipeline.py     CLIP image index (FAISS) + LLaVA VQA + RAG Stable Diffusion
|
|-- safety/                     # AI security
|   |-- adversarial_tests.py       Red-team suite (9 attack types)
|   +-- semantic_safety.py         Anomaly detection + embedding safety
|
|-- experiments/                # A/B testing
|   +-- ab_framework.py            Statistical significance framework
|
|-- mlops/                      # Evaluation + tracking
|   |-- ragas_tracker.py           Baseline capture + regression alerts
|   +-- evaluate.py                RAGAS scoring pipeline
|
|-- cicd/                       # CI/CD quality gate
|   +-- ragas_gate.py              Regression gate script (exit 1/2 blocks deploy)
|
|-- spark_ml/                   # Spark ML training pipeline
|   |-- spark_ml_pipeline.py       GBT intent classifier + KMeans clustering + Delta Lake store
|   +-- databricks_notebook.py     Databricks notebook — Unity Catalog integration + model registry
|
|-- microservices/              # Microservices architecture
|   |-- microservices.py           ServiceRegistry, EventBus (Kafka/Redis), gateway + services
|   +-- docker-compose.microservices.yml  Distributed stack (all services + Kafka + Redis)
|
|-- eval/                      # Evaluation tooling
|   +-- gradio_eval_ui.py          Gradio UI — RAG query, RAGAS, model comparison, safety tester
|
|-- config/                    # Configuration management
|   +-- hydra_config.py            Hydra structured configs + provider factory
|
|-- conf/                      # Hydra config groups (YAML)
|   |-- config.yaml                Root (assembles all groups)
|   |-- model/                     gpt4o_mini, llama3_1b, llama3_8b
|   |-- retrieval/                 faiss, hybrid, azure_search
|   |-- training/                  qlora, rlhf_ppo, dpo
|   |-- eval/                      ragas, deepeval
|   |-- infra/                     local, aws, azure
|   +-- experiment/                baseline, ablation_retrieval, ablation_reranking
|
|-- monitoring/                # Production observability stack
|   |-- prometheus_monitoring.py   LLMOpsMetrics class, instrument_app(), Grafana builder
|   |-- prometheus.yml             Scrape config (API, Celery, DCGM, Redis, node-exporter)
|   |-- alerts.yml                 SLO alert rules (p99 latency, error rate, RAGAS, NDCG, GPU)
|   +-- grafana/                   Auto-provisioned datasource + dashboard JSON
|
|-- compile/                   # torch.compile / AoT compilation pipeline
|   +-- torch_compile.py           ModelCompiler, GraphBreakDetector, AoTAutogradCompiler,
|                                  DynamicShapeManager, compile_rag_components()
|
|-- observability/             # Low-level Prometheus middleware (FastAPI)
|   |-- metrics.py                 FastAPI middleware, metric definitions, context managers
|   |-- grafana_dashboard.json     Pre-built Grafana dashboard (HTTP, inference, RAGAS)
|   |-- prometheus.yml             Prometheus scrape config
|   |-- alerts.yml                 Alerting rules (error rate, p99 latency, RAGAS drop)
|   +-- alertmanager.yml           Slack alert routing
|
|-- csrc/                      # Custom CUDA C++ extension
|   |-- llmops_kernels.cu          Fused attention clip, top-k sampling, RMSNorm kernels
|   |-- setup.py                   PyTorch CUDAExtension build script
|   +-- kernels_wrapper.py         Python wrapper with CPU fallbacks
|
|-- streaming/                 # Event-driven streaming pipeline
|   +-- streaming_pipeline.py      Kafka + Kinesis producers/consumers, DLQ, DynamoDB checkpointing
|
|-- benchmarks/                 # Performance measurement
|-- dashboard/                  # React/TypeScript monitoring UI
|-- infra/                      # Cloud + deployment
|   |-- aws_observability.py       CloudWatch dashboards + alarms
|   |-- azure_deploy.py            Azure Container Apps + Azure ML + Bicep IaC
|   |-- distributed_faiss_service  Shard + aggregator microservices
|   |-- sagemaker_model_registry   Model registry + A/B deploy
|   |-- k8s/                       Kubernetes manifests
|   |-- terraform/                 AWS infrastructure as code (Terraform)
|   +-- azure/                     Azure IaC — Bicep (main.bicep) + Terraform (main.tf)
|                                  + azure_deployment.py (AzureMLManager, AzureOpenAIBackend,
|                                    AzureContainerAppsDeployer, AzureMonitor)
|
|-- tests/                      # 66 tests (unit + integration + adversarial)
|-- docker-compose.yml          # Local monolith mode
+-- requirements.txt
```

<br>

---

## Observability & Monitoring

The system is instrumented end-to-end across three layers:

**Quality monitoring (RAGAS)**
- Faithfulness, answer relevancy, context precision, and context recall tracked on every evaluation run via MLflow
- Baseline stored in `mlops/ragas_baseline.json`; any run that drops more than 3% on any metric triggers a CI failure
- GitHub Actions posts a metric table on every PR so regressions are visible before merge

**Infrastructure monitoring (AWS CloudWatch)**
- Custom CloudWatch dashboard (`infra/aws_observability.py`) with panels for API latency (p50/p99), throughput, FAISS query time, and reranking latency
- Alarms fire on: p99 latency > 10s, error rate > 5%, FAISS query time > 100ms
- Alarm state writes to an SNS topic for downstream alerting

**Infrastructure monitoring (Azure Monitor)**
- `AzureMonitor.emit_metric()` in `infra/azure/azure_deployment.py` pushes custom metrics to the Azure Monitor REST API
- Mirrors the CloudWatch layer for the Azure deployment path
- Application Insights resource provisioned in Terraform for distributed tracing

**Task queue monitoring (Flower)**
- Celery Flower dashboard exposed on port 5555 in both Docker Compose configurations
- Tracks task success rate, retry count, and queue depth per worker

<br>

---

## Architecture Decisions

Plain-English rationale for the non-obvious choices:

**Why LangGraph instead of a simple chain?**
A chain runs top-to-bottom and stops. LangGraph is a directed graph — each node can inspect the full state, decide which node to call next, and recover from failures without restarting. That matters when retrieval returns nothing useful (route to a fallback), or when the safety check fires mid-pipeline (short-circuit before generation). The extra complexity pays off as soon as you need conditional logic.

**Why FAISS over a managed vector database?**
FAISS runs in-process — no network hop, no managed service cost, no vendor lock-in. The distributed shard design (4 shards + async fan-out aggregator) gives horizontal scale without changing the query interface. The trade-off is that FAISS does not persist across restarts without explicit save/load, and it does not support real-time updates as cleanly as Pinecone or Weaviate. For a research assistant with periodic re-indexing, that is acceptable.

**Why two-stage retrieval (bi-encoder + cross-encoder)?**
Bi-encoders (FAISS) are fast but approximate — they compare embeddings independently, missing subtle relevance signals. Cross-encoders read the query and document together, catching nuance the bi-encoder misses. Running cross-encoding on all candidates would be too slow; running it only on the top-50 FAISS results keeps end-to-end latency under 50ms while improving precision.

**Why QLoRA instead of full fine-tuning?**
Full fine-tuning an 8B model requires roughly 80GB of GPU memory. QLoRA compresses the frozen base model to 4-bit (reducing memory by roughly 4x) and trains only small LoRA adapter matrices injected into attention layers — less than 1% of total parameters. The quality gap versus full fine-tuning is small for most tasks; the hardware requirement drops from 4x A100s to a single consumer GPU.

**Why Kafka + Redis Streams (both)?**
Kafka is the right choice for production: durable, ordered, replayable, consumer groups. Redis Streams is the right choice for local development: zero infrastructure, same API shape, instant startup. The `EventBus` class auto-detects which backend to use from environment variables, so the same code runs locally and in production without changes.

**Why LangChain on top of LangGraph?**
LangGraph handles orchestration — the graph structure, state management, and routing. LangChain handles prompt engineering — versioned templates, few-shot examples, output parsers, and chain composition. They solve different problems and compose cleanly. Using both means the prompt logic is testable and swappable independently of the graph topology.

<br>

---

<div align="center">

**Built by [Joseph Ahn](https://github.com/JosephAhn23)**

</div>
