# LLMOps Research Assistant (Recruiter Version)

An end-to-end LLM platform project showing applied skills across RAG architecture, distributed retrieval, fine-tuning, MLOps, and cloud deployment.

## Portfolio Headline

Built and deployed a multi-agent RAG system with distributed FAISS retrieval, async inference, LoRA fine-tuning (Accelerate + Ray), and MLflow/RAGAS evaluation, packaged for Docker, Kubernetes, and AWS workflows.

## Why This Is Resume-Strong

- Full lifecycle ownership: ingestion -> retrieval -> reranking -> synthesis -> evaluation -> deployment
- Production-relevant architecture: FastAPI + Celery/Redis + shard microservices + observability
- Practical distributed systems patterns: local shard fan-out/merge and service-based retrieval aggregation
- Demonstrated MLOps depth: experiment tracking, quality metrics, cloud logging, and retraining scaffolds

## Tech Signals Recruiters Look For

- **LLM/RAG:** OpenAI, LangGraph, HuggingFace embeddings, cross-encoder reranking
- **Vector Search:** FAISS IVF indexing, sharding, top-k merge
- **Training:** PEFT LoRA, Accelerate, Ray Train
- **Data/ETL:** HuggingFace Datasets, Pandas, Spark pipelines, dedup + quality checks
- **MLOps:** MLflow, RAGAS, metrics/artifacts workflows
- **Infra:** Docker Compose, Kubernetes manifests, AWS (CloudWatch/S3/SageMaker), Terraform

## Copy-Paste Resume Bullets

Use these directly, then add your own measured numbers where possible.

- Architected a multi-agent RAG pipeline (Retriever -> Reranker -> Synthesizer) with typed workflow state and fault-aware routing for reliable answer generation.
- Implemented distributed FAISS retrieval via sharded indexing and global top-k aggregation, with both in-process and microservice fan-out patterns.
- Built realtime and batch inference APIs using FastAPI + Celery/Redis, enabling asynchronous job orchestration and scalable query processing.
- Developed LoRA fine-tuning workflows with both Accelerate and Ray Train to support mixed-precision, gradient accumulation, and multi-worker training.
- Added MLOps instrumentation with MLflow and RAGAS to track retrieval/generation quality metrics, run metadata, and evaluation artifacts.
- Integrated cloud-ready deployment paths using Kubernetes manifests, Terraform IaC, and AWS logging/storage/retraining primitives.

## ATS Keyword Pack

LLMOps, Retrieval-Augmented Generation, RAG, LangGraph, FastAPI, Celery, Redis, FAISS, vector search, cross-encoder reranking, PEFT, LoRA, HuggingFace, Accelerate, Ray Train, MLflow, RAGAS, Spark, Kubernetes, Docker, Terraform, AWS, CloudWatch, SageMaker.

## Interview Talking Track (30 seconds)

1. "This project proves I can build beyond prompts: I designed the full LLM system lifecycle."
2. "I implemented two distributed retrieval patterns and can explain their tradeoffs under latency and scaling constraints."
3. "I paired model adaptation (LoRA) with production concerns like observability, batch execution, and deployment."
4. "I treated evaluation as first-class using MLflow and RAGAS instead of relying on subjective output checks."

## Upgrade To 9.5/10 Recruiter Impact

To make this project stand out even more, add concrete measured outcomes to the bullets above:

- p95 latency before/after reranker changes
- retrieval hit-rate or context precision improvements
- cost/query or throughput improvements from sharding
- model quality deltas from fine-tuning (faithfulness/relevancy)
- uptime/error-rate metrics from load testing

## Quick Links

- Technical README: `README.md`
- API entrypoint: `api/main.py`
- Orchestrator: `agents/orchestrator.py`
- Distributed retrieval: `infra/distributed_index.py`
- Fine-tuning: `finetune/`
- Kubernetes: `infra/k8s/`
- Terraform: `infra/terraform/`
