# Job Requirement to Code Map

This document maps common ML engineering job requirements to the specific files and modules in this repository that demonstrate each skill. Use this during interview prep to quickly locate the relevant code.

---

## Core ML Engineering

| Requirement | File(s) | What to say |
|:---|:---|:---|
| LLM fine-tuning | `finetune/peft_lora_finetune.py` | QLoRA with 4-bit NF4, LoRA adapters on attention layers, PEFT + BitsAndBytes + Accelerate |
| RLHF / PPO | `rl/rlhf_pipeline.py` | Bradley-Terry reward model, KL-penalised policy optimisation, Process Reward Model |
| Distributed training | `finetune/ray_train_finetune.py` | Ray Train with fault-tolerant checkpointing, elastic scaling |
| Quantization | `finetune/quantization.py`, `inference/trtllm_nim.py` | INT4/INT8/FP8 via BitsAndBytes and TRT-LLM |
| Custom CUDA kernels | `cuda_ext/cuda_kernels.py`, `csrc/` | Fused softmax+temperature (1.8x speedup), RoPE, top-k sampling; PyTorch C++ extension |
| torch.compile / AoT | `compile/torch_compile.py` | Graph break detection, AoT export, dynamic shape management, MLflow benchmark logging |
| Tokenization from scratch | `tokenization/tokenization_suite.py` | BPE (GPT-2 style regex merge), WordPiece (likelihood score), multilingual fertility analysis |
| Interpretability | `interpretability/` | Attention entropy, linear probes across all layers, activation patching, CKA similarity |

---

## Production ML Systems

| Requirement | File(s) | What to say |
|:---|:---|:---|
| RAG pipeline | `agents/orchestrator.py`, `agents/retriever.py`, `agents/reranker.py`, `agents/synthesizer.py` | LangGraph stateful graph, two-stage retrieval (FAISS + cross-encoder), conditional routing |
| Vector search | `ingestion/faiss_indexer.py` | 4-shard distributed FAISS, async fan-out aggregator, sub-5ms locally |
| LLM serving | `inference/vllm_backend.py`, `inference/trtllm_nim.py` | vLLM continuous batching, TRT-LLM engine builder, NVIDIA NIM adapter |
| MoE serving | `inference/moe_serving.py` | Mixtral/DeepSeek configs, tensor + expert parallelism, expert load monitoring |
| Streaming inference | `api/websocket_handler.py` | Token streaming over WebSocket, backpressure handling |
| FastAPI gateway | `api/main.py` | Rate limiting, input validation, Prometheus middleware, health endpoints |
| Observability | `monitoring/`, `observability/` | Prometheus + Grafana, SLO alerts, CloudWatch/Azure Monitor, OpenTelemetry in multi-agent |
| CI/CD quality gate | `cicd/ragas_gate.py` | RAGAS regression blocking on PR merge, configurable thresholds |

---

## Multi-Agent Systems

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Multi-agent orchestration | `agents/multi_agent/supervisor.py` | Dynamic task decomposition, parallel agent execution, iterative refinement loop |
| Agent routing | `agents/multi_agent/routing.py` | Complexity-based, capability-based (set cover), performance-based (epsilon-greedy) |
| Consensus strategies | `agents/multi_agent/consensus.py` | Majority vote, weighted confidence, debate-based refinement |
| Failure handling | `agents/multi_agent/failure_handling.py` | Circuit breaker (CLOSED/OPEN/HALF_OPEN), exponential backoff with jitter, graceful degradation |
| Shared memory | `agents/multi_agent/memory.py` | TTL-evicted short-term store with optimistic locking (version vectors), semantic long-term retrieval |
| HITL checkpoints | `agents/multi_agent/supervisor.py` | Human-in-the-loop triggers on low confidence or safety flags |
| Agent tracing | `agents/multi_agent/supervisor.py` | OpenTelemetry span creation, structured trace logging with task_id |
| LangGraph | `agents/orchestrator.py` | Stateful graph, conditional edges, tool protocols |

---

## Experimentation and Causal Inference

| Requirement | File(s) | What to say |
|:---|:---|:---|
| A/B testing | `experimentation/ab_router.py` | Deterministic MD5 hash bucketing, MLflow observation logging, guardrail auto-stop |
| Sequential testing | `experimentation/sequential_testing.py` | O'Brien-Fleming and Pocock alpha spending, continuous monitoring without Type-I inflation |
| Variance reduction | `experimentation/cuped.py` | CUPED with pre-experiment covariates, optimal theta via covariance/variance ratio |
| Causal inference | `experimentation/double_ml.py`, `causal/causal_inference.py` | Double ML with k-fold cross-fitting, unbiased ATE from observational data |
| Uplift modeling | `causal/causal_inference.py` | T-Learner uplift model, propensity score matching, synthetic experiment simulator |
| Power analysis | `experimentation/power_analysis.py` | Sample size calculator (continuous + binary), MDE for fixed N, sensitivity analysis |
| Experiment guardrails | `experimentation/guardrails.py` | SRM detection (chi-squared), latency degradation check, covariate imbalance (SMD) |
| Experiment reporting | `experimentation/reporting.py` | Automated markdown report with CI, p-values, guardrail status, business impact recommendation |

---

## Data Engineering and Feature Stores

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Spark ML | `spark_ml/`, `ingestion/spark_pipeline.py` | GBT intent classifier, KMeans clustering, TF-IDF, Word2Vec |
| Delta Lake | `spark_ml/delta_pipeline.py` | Medallion architecture (bronze/silver/gold), schema evolution, time-travel |
| Feature store | `spark_ml/feature_store.py` | Point-in-time correct joins, online/offline split, schema versioning, feature lineage |
| MLflow model registry | `spark_ml/model_registry_flow.py` | Gated promotion (staging -> production), champion/challenger comparison, rollback |
| Databricks / Unity Catalog | `spark_ml/` | Unity Catalog integration pattern, Delta sharing, governance hooks |
| Streaming features | `streaming/realtime_features.py` | Stateful stream processor, online embedding refresh, micro-batch ingestion |
| Data quality | `streaming/realtime_features.py` | PSI distribution monitoring, Page-Hinkley drift detection |

---

## Governance and Responsible AI

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Model cards | `governance/model_card_generator.py` | Mitchell et al. standard, auto-populated from MLflow run metadata, JSON + markdown output |
| Fairness evaluation | `governance/bias_checks.py` | Statistical parity difference, equal opportunity difference, disparate impact ratio |
| CI fairness enforcement | `governance/ci_enforcement.py` | `--exit-code` flag for GitHub Actions, fails build on fairness regression above threshold |
| Audit logging | `governance/audit_log.py` | SHA-256 hash chain (simplified blockchain), tamper-proof, JSONL persistence |
| PII redaction | `governance/pii_redaction.py` | 10 pattern types (email, SSN, credit card, API keys, etc.), batch audit, configurable redaction style |
| Governance dashboard | `governance/api_router.py` | FastAPI `GET /governance/report`, model card endpoint, audit log verification |

---

## Safety and Security

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Adversarial ML | `safety_ml/adversarial_classifier.py` | GBM jailbreak classifier, 12-attack red-team library, ASR measurement |
| Prompt injection detection | `safety_ml/adversarial_classifier.py` | Embedding-based injection detector, multi-layer defense pipeline |
| Behavioral classifiers | `safety_ml/behavioral_classifiers.py` | Toxicity, intent (informational/transactional/instructional), topic router |
| Sandboxed execution | `sandbox/sandboxed_execution.py` | Docker runner with `--network none`, memory/CPU limits, static security analysis |

---

## Recommendation Systems

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Learn-to-rank | `recsys/` | LightGBM LambdaMART, feature engineering, offline NDCG/MAP/MRR evaluation |
| Explainability | `recsys/` | SHAP feature importance, per-recommendation explanation |
| Embedding retrieval | `recsys/` | Hybrid dense + sparse retrieval, MMR diversity reranking |

---

## Context Engineering

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Prompt compression | `context_engineering/context_manager.py` | Extractive compression, 35% token reduction at same RAGAS scores |
| Query rewriting | `context_engineering/context_manager.py` | HyDE, step-back prompting, sub-query decomposition, query expansion |
| Token cost optimization | `context_engineering/context_manager.py` | Token budget allocation, model routing by query complexity |
| Memory decay | `context_engineering/context_manager.py` | TTL-based context eviction, relevance-weighted retention |

---

## Infrastructure and Deployment

| Requirement | File(s) | What to say |
|:---|:---|:---|
| Docker / Kubernetes | `docker-compose.yml`, `infra/k8s/` | Multi-service Compose, K8s manifests with resource limits and health checks |
| Terraform (AWS) | `infra/terraform/` | VPC, EC2, ALB, RDS, S3, IAM -- full production stack |
| Azure IaC | `infra/azure/` | Container Apps, Bicep templates, Azure Monitor integration |
| MCP server | `mcp_server/server.py` | 6 tools over stdio transport, works with Claude Desktop and Cursor |
| Multimodal | `multimodal/` | CLIP retrieval, Stable Diffusion RAG grounding, LLaVA VQA |

---

## Interview Talking Points by Role

### ML Engineer (Applied)
Lead with: two-stage retrieval, QLoRA fine-tuning, RAGAS CI gate, A/B testing with sequential testing, causal inference for treatment effect estimation.

### ML Infrastructure Engineer
Lead with: vLLM/TRT-LLM serving, custom CUDA kernels, MoE expert parallelism, Kafka event bus, Prometheus/Grafana observability, Terraform IaC.

### Senior / Staff ML Engineer
Lead with: multi-agent system architecture (circuit breakers, consensus, HITL), Double ML causal inference, governance CI enforcement, feature store with point-in-time joins, OpenTelemetry tracing.

### ML Platform / MLOps Engineer
Lead with: MLflow model registry with gated promotion, Delta Lake medallion pipeline, RAGAS CI gate, governance dashboard, streaming drift detection, Docker/K8s deployment.

### Fintech / Enterprise ML
Lead with: SHA-256 cryptographic audit log, PII redaction with GDPR framing, fairness CI enforcement, model cards, uplift modeling for targeted feature rollout, cost-per-query analysis.
