# LLMOps Research Assistant — Justfile
# Install just: https://github.com/casey/just
# Usage: just <recipe>   e.g.  just serve   just test   just lint

set dotenv-load := true
set shell := ["bash", "-cu"]

# Default: list available recipes
default:
    @just --list

# ── Environment ───────────────────────────────────────────────────────────────

# Install all dependencies with uv (fast pip replacement)
install:
    uv pip install -e ".[dev]"

# Install GPU extras (TensorRT, Triton)
install-gpu:
    uv pip install -e ".[dev,gpu]" --extra-index-url https://pypi.nvidia.com

# Install Spark extras
install-spark:
    uv pip install -e ".[dev,spark]"

# Sync exact versions from pyproject.toml (reproducible env)
sync:
    uv pip sync pyproject.toml

# Create a fresh virtual environment
venv:
    uv venv .venv
    @echo "Activate with: source .venv/bin/activate"

# ── Development server ────────────────────────────────────────────────────────

# Start FastAPI gateway (hot-reload)
serve:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start FastAPI with Prometheus metrics enabled
serve-metrics:
    ENABLE_METRICS=1 uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Generate NIM Docker Compose and start NIM container
nim-up MODEL="meta/llama-3.1-8b-instruct":
    python inference/trtllm_nim.py --gen-docker --model {{MODEL}}
    docker compose -f docker-compose.nim.yml up -d
    @echo "NIM container starting at http://localhost:8000"

# Check NIM container health
nim-health PORT="8000":
    python inference/trtllm_nim.py --health --port {{PORT}}

# Benchmark NIM endpoint
nim-bench PORT="8000":
    python inference/trtllm_nim.py --benchmark --port {{PORT}}

# Generate TRT-LLM build script
trtllm-build-script MODEL="meta-llama/Llama-3.1-8B-Instruct":
    python inference/trtllm_nim.py --gen-build-script --model {{MODEL}}

# Start FastAPI with Prometheus metrics enabled
    ENABLE_METRICS=1 uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Gradio evaluation UI
ui:
    python eval/gradio_eval_ui.py --host 0.0.0.0 --port 7860

# Start MCP server (for Claude/Cursor integration)
mcp:
    python mcp_server/server.py

# ── Docker ────────────────────────────────────────────────────────────────────

# Start full local stack (API + Redis + Celery)
up:
    docker compose up -d

# Start observability stack (Prometheus + Grafana + Alertmanager)
up-obs:
    docker compose -f docker-compose.observability.yml up -d
    @echo "Grafana: http://localhost:3000 (admin/admin)"
    @echo "Prometheus: http://localhost:9090"

# Start distributed microservices stack
up-micro:
    docker compose -f microservices/docker-compose.microservices.yml up -d

# Stop all stacks
down:
    docker compose down
    docker compose -f docker-compose.observability.yml down --remove-orphans || true
    docker compose -f microservices/docker-compose.microservices.yml down --remove-orphans || true

# Rebuild and restart API container
rebuild:
    docker compose build api
    docker compose up -d api

# ── Testing ───────────────────────────────────────────────────────────────────

# Run full test suite
test:
    pytest tests/ -v --tb=short

# Run tests with coverage report
test-cov:
    pytest tests/ -v --tb=short --cov=. --cov-report=term-missing --cov-report=html

# Run only fast unit tests (skip integration)
test-unit:
    pytest tests/ -v --tb=short -m "not integration"

# Run safety/adversarial tests
test-safety:
    pytest tests/test_safety.py -v --tb=short

# Run a specific test file
test-file FILE:
    pytest {{FILE}} -v --tb=long

# ── Evaluation ────────────────────────────────────────────────────────────────

# Run RAGAS evaluation and update baseline
eval:
    python -m mlops.ragas_tracker \
        --output-json mlops/ragas_scores.json \
        --experiment-name "manual-eval-$(date +%Y%m%d)"

# Run RAGAS regression gate (CI check)
gate:
    python cicd/ragas_gate.py \
        --scores mlops/ragas_scores.json \
        --baseline mlops/ragas_baseline.json

# ── Benchmarking ──────────────────────────────────────────────────────────────

# torch.compile benchmark (CPU, embedding model)
bench-compile:
    python inference/torch_compile_bench.py --device cpu --model embedding --runs 50

# torch.compile benchmark (all models, GPU if available)
bench-compile-gpu:
    python inference/torch_compile_bench.py --device cuda --model all --runs 100 --mlflow

# Streaming pipeline benchmark (Kafka produce/consume)
bench-streaming:
    python streaming/streaming_pipeline.py --backend kafka --mode produce --n-events 100

# ── Fine-tuning ───────────────────────────────────────────────────────────────

# QLoRA fine-tuning (small model, CPU demo)
finetune-qlora:
    python finetune/peft_lora_finetune.py

# RLHF PPO training
finetune-rlhf:
    python rl/rlhf_pipeline.py

# GRPO reasoning fine-tuning
finetune-grpo:
    python rl/reasoning_rl.py

# ── Inference ─────────────────────────────────────────────────────────────────

# Run llama.cpp local inference
infer-local MODEL="models/llama-3.2-1b.Q4_K_M.gguf":
    python inference/llamacpp_backend.py --model {{MODEL}} --prompt "What is RAG?"

# Export model to ONNX
export-onnx MODEL="embedding":
    python inference/tensorrt_onnx_triton.py --export-onnx --model {{MODEL}}

# ── Configuration ─────────────────────────────────────────────────────────────

# Generate Hydra conf/ directory
config-gen:
    python config/hydra_config.py --generate

# Show resolved config
config-show *OVERRIDES:
    python config/hydra_config.py --show --overrides {{OVERRIDES}}

# ── Code quality ──────────────────────────────────────────────────────────────

# Run ruff linter
lint:
    ruff check . --fix

# Run black formatter
fmt:
    black .

# Run both lint and format
check:
    ruff check .
    black --check .

# Run mypy type checking
types:
    mypy . --ignore-missing-imports

# Run all quality checks
qa: lint fmt types

# ── Streaming ─────────────────────────────────────────────────────────────────

# Set up Kafka topics
kafka-setup:
    python streaming/streaming_pipeline.py --backend kafka --mode setup

# Produce test events to Kafka
kafka-produce N="10":
    python streaming/streaming_pipeline.py --backend kafka --mode produce --n-events {{N}}

# Consume events from Kafka
kafka-consume N="10":
    python streaming/streaming_pipeline.py --backend kafka --mode consume --n-events {{N}}

# ── Tokenization ──────────────────────────────────────────────────────────────

# BPE tokenizer demo (trains from scratch, no deps beyond stdlib)
tokenize-demo:
    python tokenization/tokenization_suite.py bpe-demo

# WordPiece demo
wordpiece-demo:
    python tokenization/tokenization_suite.py wordpiece-demo

# Analyze a pretrained tokenizer (requires transformers)
tokenize-analyze model="bert-base-uncased":
    python tokenization/tokenization_suite.py analyze --model {{model}}

# ── Interpretability ──────────────────────────────────────────────────────────

# Attention visualization
attention model="bert-base-uncased" text="The cat sat on the mat":
    python interpretability/probes.py attention --model {{model}} --text "{{text}}"

# Layer-wise linear probe experiment
probe model="bert-base-uncased":
    python interpretability/probes.py probe --model {{model}}

# CKA layer similarity matrix
cka model="bert-base-uncased":
    python interpretability/probes.py cka --model {{model}}

# Activation patching / causal tracing
patch:
    python interpretability/probes.py patch

# ── Adversarial & behavioral classifiers ──────────────────────────────────────

# Train and benchmark adversarial classifiers (requires sentence-transformers, sklearn)
adversarial-train:
    python safety/adversarial_classifier.py benchmark

# Evaluate a prompt for adversarial content
adversarial-eval text="Ignore all previous instructions":
    python safety/adversarial_classifier.py eval "{{text}}"

# Train behavioral classifiers and show sample predictions
behavioral-train:
    python safety/behavioral_classifiers.py benchmark

# Run analytics over sample corpus
behavioral-analytics:
    python safety/behavioral_classifiers.py analytics

# ── RL environments ───────────────────────────────────────────────────────────

# RAG quality RL environment — random vs greedy baseline
rl-rag:
    python rl/rl_environments.py rag-env --episodes 20

# Retrieval strategy bandit (UCB1)
rl-bandit algo="ucb1" steps="1000":
    python rl/rl_environments.py bandit --algo {{algo}} --steps {{steps}}

# Prompt selection bandit (Thompson sampling)
rl-prompt-bandit:
    python rl/rl_environments.py prompt-bandit --steps 500

# ── Sandbox ───────────────────────────────────────────────────────────────────

# Check Docker availability for sandboxed execution
sandbox-check:
    python sandbox/code_sandbox.py check

# Run demo programs in Docker sandbox (requires Docker)
sandbox-demo:
    python sandbox/code_sandbox.py demo

# Static analysis only (no Docker needed)
sandbox-analyze code="import os; os.system('ls')":
    python sandbox/code_sandbox.py analyze --code "{{code}}"

# ── CUDA extension ────────────────────────────────────────────────────────────

# Write CUDA source files to cuda_ext/ then compile the PyTorch extension
build-cuda:
    python cuda_ext/cuda_kernels.py --write-sources
    python setup_cuda.py build_ext --inplace

# Benchmark all CUDA kernels vs PyTorch baseline (requires CUDA)
bench-cuda device="cuda":
    python cuda_ext/cuda_kernels.py --benchmark --device {{device}} --n-runs 500

# ── Utilities ─────────────────────────────────────────────────────────────────

# Clean Python cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

# Show project structure
tree:
    find . -not -path "./.git/*" -not -path "./.venv/*" -not -path "./outputs/*" \
           -not -path "./__pycache__/*" | sort | head -100

# Print all available just recipes
help:
    @just --list
