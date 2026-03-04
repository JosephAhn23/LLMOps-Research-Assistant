<div align="center">

# LLMOps Research Assistant

**Production-architecture AI platform covering the full LLMOps lifecycle.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)

*Not a wrapper. A production system built from raw components — distributed vector search, quantized fine-tuning, RLHF/PPO, multimodal generation, streaming inference, safety layers, and multi-cloud deployment.*

</div>

---

## What It Does

| Stage | Implementation | Runs |
|:---|:---|:---|
| **Ingest** | Chunking → embedding → FAISS indexing. HuggingFace Datasets + MinHash dedup. CommonCrawl WARC parsing (S3). Spark distributed ingestion + Spark ML feature engineering (TF-IDF, Word2Vec, K-Means). | Core: ✅ locally. WARC/Spark: requires S3 + cluster. |
| **Search** | Two-stage: FAISS bi-encoder scan across 4 distributed shards, then cross-encoder reranking. Sub-50ms locally. | ✅ Locally (single-node + 4-shard Docker Compose). |
| **Generate** | LangGraph agentic pipeline — stateful graph (retriever → reranker → synthesizer), conditional routing, tool protocols. Token streaming over WebSocket. vLLM backend for self-hosted inference. | ✅ Locally with GPT-4o-mini. vLLM requires GPU. |
| **Fine-tune** | QLoRA (4-bit NF4) via PEFT + BitsAndBytes + Accelerate. RLHF with PPO — Bradley-Terry reward model, KL-penalised policy optimisation, Process Reward Model. GRPO reasoning fine-tuning (DeepSeek-R1/VeRL style). Ray Train for fault-tolerant distributed training. | Requires GPU. Not run in CI. |
| **Evaluate** | RAGAS (faithfulness, relevancy, precision, recall) with MLflow tracking. **Wired into GitHub Actions CI** — quality gate blocks merge on regression > 5%. | ✅ Locally with OpenAI key. CI gate runs on every PR. |
| **Serve** | TensorRT-LLM engine builder (FP16/INT8/FP8), ONNX/Triton pipeline, NVIDIA NIM adapter, MoE expert parallelism (Mixtral/DeepSeek), custom CUDA kernels (fused attention, RMSNorm, top-k). | GPU/A100 required for TRT-LLM. ONNX path runs on CPU. |
| **Secure** | Rule-based injection detection, embedding anomaly detection, LLM-as-judge red-team suite (9 attack categories), ML jailbreak classifier, behavioral classifiers (toxicity, intent, topic). | ✅ Locally. |
| **Deploy** | Docker Compose, Kubernetes manifests, Terraform (AWS: VPC/EC2/ALB/RDS), Azure Container Apps + Bicep IaC. | Docker Compose: ✅ locally. K8s/Terraform: implemented, not live. |
| **Connect** | MCP server (stdio) exposes retrieve, ingest, evaluate, and benchmark as tools for Claude Desktop / Cursor. | ✅ Locally. |

---

## Performance

| Metric | Value | Notes |
|:---|:---|:---|
| Vector search latency | `< 5 ms` | Single-node FAISS, measured locally |
| Reranking latency | `~40 ms` | Cross-encoder on CPU, measured locally |
| End-to-end p50 | `3,284 ms` | Includes GPT-4o-mini API round-trip |
| End-to-end p99 | `6,238 ms` | Includes GPT-4o-mini API round-trip |
| Throughput | `0.9 QPS` | Single node, sequential — bottleneck is the external LLM API call, not the retrieval stack. Parallelising requests or switching to a local vLLM backend removes this ceiling. |
| vLLM fp16 | `~1,500 tok/s` | Architecture target based on published A100 benchmarks |
| vLLM int4-AWQ | `~3,000 tok/s` | Architecture target based on published A100 benchmarks |

### RAGAS Quality Scores

| Metric | Score |
|:---|:---|
| Faithfulness | **0.847** |
| Answer Relevancy | **0.823** |
| Context Precision | **0.791** |
| Context Recall | **0.812** |

> Measured on a held-out evaluation set using GPT-4o-mini as both synthesis and judge model.

---

## Architecture Decisions

**Why LangGraph instead of a simple chain?**
A chain runs top-to-bottom and stops. LangGraph is a directed graph — each node can inspect the full state, decide which node to call next, and recover from failures without restarting. That matters when retrieval returns nothing useful (route to fallback) or when the safety check fires mid-pipeline (short-circuit before generation).

**Why two-stage retrieval (bi-encoder + cross-encoder)?**
Bi-encoders (FAISS) are fast but approximate — they compare embeddings independently, missing subtle relevance signals. Cross-encoders read the query and document together, catching nuance the bi-encoder misses. Running cross-encoding on all candidates would be too slow; running it only on the top-50 FAISS results keeps end-to-end latency under 50ms while improving precision.

**Why FAISS over a managed vector database?**
FAISS runs in-process — no network hop, no managed service cost, no vendor lock-in. The distributed shard design (4 shards + async fan-out aggregator) gives horizontal scale without changing the query interface. The trade-off: no real-time updates as cleanly as Pinecone or Weaviate. For a research assistant with periodic re-indexing, that's acceptable.

**Why QLoRA instead of full fine-tuning?**
Full fine-tuning an 8B model requires ~80GB of GPU memory. QLoRA compresses the frozen base model to 4-bit and trains only small LoRA adapter matrices injected into attention layers — less than 1% of total parameters. The quality gap versus full fine-tuning is small for most tasks; the hardware requirement drops from 4× A100s to a single consumer GPU.

**Why Kafka + Redis Streams (both)?**
Kafka is the right choice for production: durable, ordered, replayable, consumer groups. Redis Streams is the right choice for local development: zero infrastructure, same API shape, instant startup. The `EventBus` class auto-detects which backend to use from environment variables — same code runs locally and in production without changes.

---

## Quick Start

```bash
git clone https://github.com/JosephAhn23/LLMOps-Research-Assistant
cd LLMOps-Research-Assistant
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# Start services + API
docker compose up -d
uvicorn api.main:app --reload

# Run quality evaluation
python -m mlops.ragas_tracker

# Fine-tune (QLoRA, requires GPU)
python -m finetune.peft_lora_finetune

# RLHF/PPO training (requires GPU)
python rl/rlhf_pipeline.py

# Local inference with llama.cpp
python inference/llamacpp_backend.py --prompt "Explain RAG"

# torch.compile benchmark (CPU)
python compile/torch_compile.py --model prajjwal1/bert-tiny --device cpu --graph-breaks

# Launch Gradio eval UI
python eval/gradio_eval_ui.py

# Start MCP server (for Claude Desktop / Cursor)
python mcp_server/server.py
```

### MCP Integration

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

### Testing

```bash
pytest                          # 66 tests (unit + integration + adversarial)
pytest tests/test_safety.py -v  # Safety + red-team tests
```

---

## Project Structure

```
agents/          LangGraph pipeline — orchestrator, retriever, reranker, synthesizer
api/             FastAPI gateway, WebSocket streaming, Celery batch queue
cicd/            RAGAS regression gate (blocks CI on quality drop)
compile/         torch.compile benchmarking, AoT export, graph break detection
config/          Hydra structured configs + provider factory
csrc/            Custom CUDA kernels — fused attention, RMSNorm, top-k sampling
cuda_ext/        Fused softmax+temperature, RoPE, top-p sampling kernels
eval/            Gradio evaluation UI
experiments/     A/B framework with statistical significance testing
finetune/        QLoRA, RLHF/PPO, GRPO, Ray fault-tolerant training, quantization
inference/       vLLM, llama.cpp, TRT-LLM, ONNX/Triton, NIM, MoE serving
ingestion/       Chunking, FAISS indexing, WARC parsing, Spark ML pipelines
interpretability/ Attention visualization, linear probes, activation patching, CKA
mcp_server/      MCP protocol server (6 tools, stdio transport)
microservices/   ServiceRegistry, EventBus (Kafka/Redis), API gateway pattern
mlops/           RAGAS tracking, MLflow integration, evaluation pipeline
monitoring/      Prometheus + Grafana stack, SLO alert rules, CloudWatch/Azure Monitor
multimodal/      CLIP retrieval, Stable Diffusion RAG grounding, LLaVA VQA
observability/   FastAPI Prometheus middleware, pre-built Grafana dashboard
rl/              RLHF pipeline (TRL), GRPO reasoning fine-tuning
safety/          Adversarial tests, semantic safety, ML classifiers, behavioral classifiers
sandbox/         Docker-based sandboxed code execution with static analysis
spark_ml/        GBT intent classifier, KMeans clustering, Databricks Unity Catalog
streaming/       Kafka + Kinesis producers/consumers, DLQ, DynamoDB checkpointing
tokenization/    BPE/WordPiece from scratch, SentencePiece, multilingual analysis
infra/           Kubernetes, Terraform (AWS), Azure Bicep/Terraform, SageMaker
tests/           66 tests — unit, integration, adversarial, shard failure modes
```

---

<div align="center">

**Built by [Joseph Ahn](https://github.com/JosephAhn23)**

</div>
