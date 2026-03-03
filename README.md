<div align="center">

<img src="assets/hero_banner.png" alt="LLMOps Research Assistant" width="100%">

# LLMOps Research Assistant

**An end-to-end AI platform that ingests, retrieves, generates, fine-tunes, evaluates, and deploys — all in one system.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)

---

*Not a wrapper. Not a tutorial. A working system built from raw components — distributed vector search, quantized fine-tuning, streaming inference, safety layers, and cloud-native deployment.*

</div>

<br>

## What This Does

Most AI projects stop at "call the API and print the answer." This one doesn't.

This platform handles the **full lifecycle** — from raw web data to a deployed, monitored, self-evaluating AI system:

| Stage | What Happens |
|---|---|
| **Ingest** | Pulls data from CommonCrawl, HuggingFace Hub, or local files. Deduplicates, filters for quality, and tokenizes at scale. |
| **Search** | Two-stage retrieval — fast approximate search across distributed shards, then precision reranking with a cross-encoder. Sub-50ms total. |
| **Generate** | Multi-agent pipeline orchestrates retrieval, reranking, and synthesis. Streams tokens in real-time over WebSocket. |
| **Fine-tune** | QLoRA training on consumer hardware — 4-bit quantization keeps a 8B model under 16GB VRAM. Ships LoRA adapters, not full checkpoints. |
| **Evaluate** | Automated quality scoring (faithfulness, relevancy, precision, recall) with regression alerts if scores drop below baseline. |
| **Secure** | Prompt injection detection, adversarial red-teaming, anomaly detection. Catches attacks before they reach the model. |
| **Deploy** | Docker Compose for dev, Kubernetes for prod, Terraform for infrastructure. SageMaker model registry with approval gates. |
| **Connect** | MCP server lets any AI agent (Claude, Cursor) use the pipeline as a tool — retrieve, ingest, evaluate, all via natural language. |

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
      +---> CloudWatch dashboards + alarms in production
```

<br>

## Performance

<div align="center">
<img src="assets/metrics_dashboard.png" alt="Performance Metrics" width="90%">
</div>

<br>

<div align="center">

| Metric | Measured |
|:---|:---|
| **Vector search latency** | `< 5 ms` |
| **Reranking latency** | `~40 ms` |
| **End-to-end p50** | `3,284 ms` |
| **End-to-end p99** | `6,238 ms` |
| **Throughput** | `0.9 QPS` (single node, incl. LLM call) |
| **vLLM fp16** | `~1,500 tok/s` (A100 target) |
| **vLLM int4-AWQ** | `~3,000 tok/s` (A100 target) |

</div>

> **Note:** End-to-end latency includes the LLM generation call (GPT-4o-mini). The retrieval + reranking pipeline alone completes in under 50ms.

### Quality Scores (RAGAS)

<div align="center">

| Metric | Score |
|:---|:---|
| Faithfulness | **0.847** |
| Answer Relevancy | **0.823** |
| Context Precision | **0.791** |
| Context Recall | **0.812** |

</div>

<br>

## Key Capabilities

### Intelligent Retrieval
Two-stage search that balances speed and accuracy. The first pass uses a bi-encoder to scan millions of chunks in milliseconds via FAISS. The second pass runs a cross-encoder that reads each query-document pair together, catching subtle relevance that embedding similarity misses.

### Quantized Fine-Tuning (QLoRA)
Train an 8-billion parameter model on a single GPU. 4-bit quantization compresses the base model while LoRA adapters inject trainable parameters into attention layers — less than 1% of total weights. Merges cleanly for zero-overhead inference.

### Real-Time Streaming
WebSocket endpoint streams tokens as they're generated. Clients see the response build word-by-word with live latency and throughput stats. Supports concurrent sessions with connection management.

### Safety & Red-Teaming
Multi-layered defense against adversarial inputs. Rule-based pattern matching for known attack signatures. Embedding-based anomaly detection for novel attacks. LLM-as-judge for ambiguous cases. Full red-team test suite with 9 attack categories.

### Data Pipelines at Scale
Ingests from CommonCrawl WARCs, HuggingFace Hub, or local files. Language detection, quality scoring, MinHash deduplication, and tokenizer-aware filtering. Spark support for distributed processing. Outputs Arrow format for fast training.

### Self-Evaluating Pipeline
RAGAS metrics run automatically after each deployment. Baseline capture, regression detection with configurable thresholds, and historical trend tracking. If quality drops, you know immediately — not after users complain.

### MCP Integration
Exposes the entire pipeline as tools via the Model Context Protocol. Any MCP-compatible agent (Claude Desktop, Cursor, VS Code) can retrieve documents, ingest new data, run evaluations, and check model status — all through natural language.

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
| **Vector Store** | FAISS IVFFlat (distributed, 4 shards) |
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

**Kubernetes:**
```bash
bash infra/k8s/deploy.sh
```

**Terraform (AWS):**
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
