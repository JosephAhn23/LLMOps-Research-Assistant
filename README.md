<div align="center">

<img src="assets/hero_banner.png" alt="LLMOps Research Assistant" width="100%">

# LLMOps Research Assistant

**A production-architecture AI platform covering the full LLMOps lifecycle: ingestion, retrieval, generation, fine-tuning, evaluation, safety, and deployment.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)

---

*Not a wrapper. Not a tutorial. A production-architecture system built from raw components — distributed vector search, quantized fine-tuning, streaming inference, safety layers, and cloud-native deployment. Core pipeline runs locally; cloud and GPU components are fully implemented and ready to deploy.*

</div>

<br>

## What This Does

Most AI projects stop at "call the API and print the answer." This one doesn't.

This platform handles the **full lifecycle** — from raw web data to a deployed, monitored, self-evaluating AI system:

| Stage | What's Implemented | Status |
|---|---|---|
| **Ingest** | Chunking, embedding, FAISS indexing from local files or HuggingFace Hub. WARC and Spark pipelines for CommonCrawl at scale. | Core pipeline: ✅ runs locally. WARC/Spark: implemented, requires S3 + cluster. |
| **Search** | Two-stage retrieval — FAISS bi-encoder scan across distributed shards, then cross-encoder reranking. Sub-50ms locally. | ✅ Runs locally (single-node and 4-shard Docker Compose). |
| **Generate** | LangGraph multi-agent pipeline (retrieve → rerank → synthesize). Token streaming over WebSocket. Default backend: GPT-4o-mini. vLLM backend implemented for self-hosted inference. | ✅ Runs locally with GPT-4o-mini. vLLM requires a GPU server. |
| **Fine-tune** | QLoRA (4-bit NF4) via PEFT + BitsAndBytes. Multi-GPU via Accelerate. Fault-tolerant distributed training via Ray Train. | Implemented. Requires a GPU (single or multi). Not run in CI. |
| **Evaluate** | RAGAS scoring (faithfulness, relevancy, precision, recall) with MLflow tracking and regression alerts. | ✅ Runs locally with an OpenAI key. |
| **Secure** | Rule-based prompt injection detection, embedding-based anomaly detection, LLM-as-judge red-team suite. | ✅ Runs locally. |
| **Deploy** | Docker Compose (local), Kubernetes manifests, Terraform for AWS (VPC, EC2, ALB, RDS, S3). SageMaker model registry with approval gates. | Docker Compose: ✅ runs locally. K8s/Terraform/SageMaker: implemented, not deployed to live AWS. |
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

### Quantized Fine-Tuning (QLoRA)
Implemented training pipeline for an 8-billion parameter model on a single GPU. 4-bit quantization compresses the base model while LoRA adapters inject trainable parameters into attention layers — less than 1% of total weights. Multi-GPU via Accelerate and fault-tolerant distributed training via Ray Train are also implemented. Requires a GPU to run.

### Real-Time Streaming
WebSocket endpoint streams tokens as they're generated. Clients see the response build word-by-word with live latency and throughput stats. Supports concurrent sessions with connection management.

### Safety & Red-Teaming
Multi-layered defense against adversarial inputs. Rule-based pattern matching for known attack signatures. Embedding-based anomaly detection for novel attacks. LLM-as-judge for ambiguous cases. Full red-team test suite with 9 attack categories.

### Data Pipelines at Scale
Ingestion from local files and HuggingFace Hub runs locally. CommonCrawl WARC parsing (S3-backed) and Apache Spark distributed processing are fully implemented pipelines — they require an S3 bucket and a Spark cluster to run at scale. All pipelines include language detection, quality scoring, MinHash deduplication, and tokenizer-aware filtering.

### Self-Evaluating Pipeline
RAGAS metrics run automatically after each deployment. Baseline capture, regression detection with configurable thresholds, and historical trend tracking. If quality drops, you know immediately — not after users complain.

### MCP Integration
Exposes the pipeline as tools via the Model Context Protocol. The MCP server is implemented and runs locally — connect it to Claude Desktop or Cursor using the config below and it will call your local pipeline directly. Retrieve documents, ingest new data, and run evaluations through natural language.

### A/B Experimentation
Built-in experiment framework with deterministic user assignment, statistical significance testing, and effect size measurement. Compare retrieval strategies, reranking models, or generation configs with confidence.

<br>

## Built With

<div align="center">

| | |
|:---|:---|
| **Models** | GPT-4o-mini, Llama-3.1-8B, Stable Diffusion v1.5 |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace) |
| **Reranker** | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| **Fine-tuning** | QLoRA via PEFT + BitsAndBytes + Accelerate |
| **Vector Store** | FAISS IndexFlatIP (distributed, 4 shards) |
| **Orchestration** | LangGraph multi-agent pipeline |
| **API** | FastAPI + WebSockets |
| **Queue** | Celery + Redis + Flower |
| **Tracking** | MLflow + RAGAS |
| **Cloud** | AWS (S3, CloudWatch, SageMaker) |
| **Agent Protocol** | MCP server (stdio transport) |
| **Containers** | Docker Compose + Kubernetes |
| **Infrastructure** | Terraform |
| **Data** | HF Datasets, Apache Spark, CommonCrawl |
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

# Start services
docker compose up -d

# Launch the API
uvicorn api.main:app --reload

# Fine-tune a model (QLoRA)
python -m finetune.peft_lora_finetune

# Run quality evaluation
python -m mlops.ragas_tracker

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
|   +-- spark_pipeline.py          Spark distributed processing
|
|-- finetune/                   # Model training
|   |-- peft_lora_finetune.py      QLoRA (4-bit) + PEFT + Accelerate
|   |-- ray_fault_tolerant.py      Ray distributed + fault recovery
|   +-- accelerate_finetune.py     Multi-GPU LoRA
|
|-- inference/                  # Model serving
|   |-- vllm_backend.py            vLLM inference backend
|   +-- diffusion_pipeline.py      Stable Diffusion + RAG grounding
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
|-- benchmarks/                 # Performance measurement
|-- dashboard/                  # React/TypeScript monitoring UI
|-- infra/                      # Cloud + deployment
|   |-- aws_observability.py       CloudWatch dashboards + alarms
|   |-- distributed_faiss_service  Shard + aggregator microservices
|   |-- sagemaker_model_registry   Model registry + A/B deploy
|   |-- k8s/                       Kubernetes manifests
|   +-- terraform/                 AWS infrastructure as code
|
|-- tests/                      # 66 tests (unit + integration + adversarial)
|-- docker-compose.yml
+-- requirements.txt
```

<br>

---

<div align="center">

**Built by [Joseph Ahn](https://github.com/JosephAhn23)**

</div>
