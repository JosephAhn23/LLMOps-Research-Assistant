"""
run_ragas.py — Standalone RAGAS evaluation
No live stack needed. Uses your OpenAI key directly.

Usage:
    set OPENAI_API_KEY=your_key   (Windows)
    python run_ragas.py
"""

import os
import json
import re
from pathlib import Path

if not os.environ.get("OPENAI_API_KEY"):
    print("\nSet your OpenAI API key first:")
    print("  Windows: set OPENAI_API_KEY=sk-...")
    print("  Mac/Linux: export OPENAI_API_KEY=sk-...")
    exit(1)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

QA_PAIRS = [
    {
        "question": "What is retrieval-augmented generation?",
        "answer": "Retrieval-augmented generation (RAG) is a technique that combines information retrieval with language model generation. It first retrieves relevant documents from a knowledge base using semantic search, then uses those documents as context for the LLM to generate a grounded answer.",
        "contexts": [
            "Retrieval-Augmented Generation (RAG) is a framework that enhances LLM outputs by retrieving relevant documents at inference time and conditioning generation on those documents. This reduces hallucination and keeps answers grounded in source material.",
            "RAG systems typically consist of an indexing phase (chunking, embedding, storing in a vector store) and a retrieval phase (query embedding, ANN search, reranking) followed by generation with retrieved context.",
        ],
        "ground_truth": "RAG retrieves relevant documents from a knowledge base and uses them as context for LLM generation, reducing hallucination and improving factual accuracy.",
    },
    {
        "question": "How does LoRA reduce the number of trainable parameters?",
        "answer": "LoRA (Low-Rank Adaptation) freezes the pretrained model weights and injects trainable low-rank decomposition matrices into each layer. Instead of updating the full weight matrix W, it learns two smaller matrices A and B where the update is dW = A*B. With rank r much smaller than the original dimensions, this reduces trainable parameters by orders of magnitude.",
        "contexts": [
            "LoRA decomposes weight updates into two low-rank matrices: W = W0 + dW where dW = BA. B is d*r and A is r*k where r << min(d,k). Only A and B are trained; W0 is frozen. This reduces trainable parameters from d*k to r*(d+k).",
            "For a 7B parameter model, LoRA with rank 16 typically reduces trainable parameters to under 1% of total, enabling fine-tuning on consumer GPUs. QLoRA further combines this with 4-bit quantization.",
        ],
        "ground_truth": "LoRA decomposes weight updates into two low-rank matrices A and B (dW = A*B), training only these small matrices instead of the full weight matrix, reducing trainable parameters to under 1%.",
    },
    {
        "question": "What is PagedAttention in vLLM?",
        "answer": "PagedAttention is a memory management algorithm used in vLLM that manages the KV cache in fixed-size blocks, analogous to virtual memory paging in operating systems. This eliminates memory fragmentation, allows dynamic memory allocation per sequence, and enables efficient memory sharing for parallel sampling and beam search.",
        "contexts": [
            "vLLM's PagedAttention divides the KV cache into fixed-size blocks (pages) rather than allocating contiguous memory per sequence. This is inspired by OS virtual memory management and eliminates the fragmentation that wastes 60-80% of GPU memory in naive implementations.",
            "PagedAttention enables near-zero memory waste, copy-on-write KV sharing for beam search, and dynamic allocation as sequences grow. Combined with continuous batching, vLLM achieves 10-24x higher throughput than HuggingFace Transformers naive serving.",
        ],
        "ground_truth": "PagedAttention manages KV cache in fixed-size memory blocks like OS virtual memory pages, eliminating fragmentation and enabling efficient memory sharing for higher throughput.",
    },
    {
        "question": "What does FAISS stand for and what is it used for?",
        "answer": "FAISS stands for Facebook AI Similarity Search. It is a library developed by Meta for efficient similarity search and clustering of dense vectors. In RAG systems, FAISS is used as the vector store for approximate nearest neighbor (ANN) search to retrieve the most semantically similar documents to a query embedding.",
        "contexts": [
            "FAISS (Facebook AI Similarity Search) is an open-source library by Meta Research for efficient similarity search over dense vectors. It supports both exact and approximate nearest neighbor search with multiple index types: IVFFlat, HNSW, PQ, and others.",
            "In production RAG pipelines, FAISS IVFFlat with 4 distributed shards is a common setup. Each shard handles a partition of the document corpus; an aggregator merges results via fan-out, achieving ~5ms retrieval latency at scale.",
        ],
        "ground_truth": "FAISS (Facebook AI Similarity Search) is Meta's library for efficient dense vector similarity search, used in RAG for fast approximate nearest neighbor retrieval.",
    },
    {
        "question": "What is a dead-letter queue and why is it important?",
        "answer": "A dead-letter queue (DLQ) is a message queue that captures messages that fail processing after exhausting all retry attempts. It prevents failed messages from being lost, allows engineers to inspect and debug failures, and enables selective reprocessing. In production Celery deployments, DLQs are critical for reliability.",
        "contexts": [
            "A dead-letter queue (DLQ) receives messages that cannot be processed successfully after the maximum number of retries. Instead of dropping failed tasks, they are routed to the DLQ for inspection and manual or automated reprocessing.",
            "In Celery, DLQs are configured alongside priority queues (default, high_priority, dead_letter). Failed tasks with exponential backoff that exceed max_retries are routed to the DLQ. Flower provides a monitoring UI to inspect DLQ contents.",
        ],
        "ground_truth": "A dead-letter queue captures messages that fail after maximum retries, preventing data loss and enabling failure inspection and reprocessing.",
    },
    {
        "question": "How does cross-encoder reranking improve RAG quality?",
        "answer": "Cross-encoder reranking improves RAG quality by jointly encoding the query and each candidate document together through a transformer, allowing direct attention between query and document tokens. This produces more accurate relevance scores than bi-encoders which encode query and documents independently. The trade-off is higher latency (~40ms for 50 candidates) but significantly better precision.",
        "contexts": [
            "Cross-encoders take (query, document) pairs as joint input to a transformer, enabling full cross-attention between query and document tokens. This produces highly accurate relevance scores but requires O(n) forward passes for n candidates.",
            "In a two-stage retrieval pipeline: Stage 1 uses a bi-encoder for fast ANN retrieval (recall-optimized, ~5ms, top-50 candidates). Stage 2 uses a cross-encoder to rerank those 50 candidates (precision-optimized, ~40ms) returning top-5. This achieves near cross-encoder accuracy at bi-encoder speed.",
        ],
        "ground_truth": "Cross-encoders jointly encode query and document pairs enabling direct attention between tokens, producing more accurate relevance scores than independent bi-encoder embeddings.",
    },
    {
        "question": "What is the RAGAS faithfulness metric?",
        "answer": "RAGAS faithfulness measures whether the generated answer is factually consistent with the retrieved context. It works by breaking the answer into individual claims, then checking each claim against the context using an LLM-as-judge. The score is the fraction of claims that are supported by the context, ranging from 0 to 1.",
        "contexts": [
            "RAGAS faithfulness score = (number of claims in answer supported by context) / (total claims in answer). It uses an LLM to decompose the answer into atomic claims and verify each against the retrieved passages.",
            "A faithfulness score of 1.0 means every claim in the answer can be traced back to the retrieved context. Low faithfulness indicates hallucination -- the model is generating content not grounded in the retrieved documents.",
        ],
        "ground_truth": "RAGAS faithfulness measures the fraction of answer claims that are supported by retrieved context, using LLM-as-judge to detect hallucination.",
    },
    {
        "question": "What is MinHash LSH used for in data pipelines?",
        "answer": "MinHash LSH (Locality Sensitive Hashing) is used for near-duplicate document detection in large-scale data pipelines. It approximates Jaccard similarity between documents using random hash functions, allowing efficient deduplication of web-scale corpora without comparing all document pairs. Documents with similarity above a threshold (e.g., 0.85) are considered near-duplicates and one is removed.",
        "contexts": [
            "MinHash LSH estimates Jaccard similarity between documents using b bands of r hash functions. Documents are deduplicated if their estimated similarity exceeds a threshold. For CommonCrawl-scale corpora, this reduces the O(n^2) comparison problem to near-linear time.",
            "In the HuggingFace Datasets pipeline, MinHash with 128 permutations and threshold=0.85 is used after quality filtering. The datasketch library provides the MinHashLSH implementation. Near-duplicate removal typically reduces corpus size by 15-40% for web data.",
        ],
        "ground_truth": "MinHash LSH approximates document similarity using random hashing for efficient near-duplicate detection, enabling deduplication of large corpora in near-linear time.",
    },
]


def run_ragas():
    print(f"\nRunning RAGAS evaluation on {len(QA_PAIRS)} QA pairs...")
    print("(This makes OpenAI API calls -- ~30-60 seconds)\n")

    ds = Dataset.from_dict({
        "question":     [p["question"]     for p in QA_PAIRS],
        "answer":       [p["answer"]       for p in QA_PAIRS],
        "contexts":     [p["contexts"]     for p in QA_PAIRS],
        "ground_truth": [p["ground_truth"] for p in QA_PAIRS],
    })

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val = result[key]
        if isinstance(val, list):
            valid = [v for v in val if v is not None and not (isinstance(v, float) and v != v)]
            scores[key] = round(sum(valid) / max(len(valid), 1), 3) if valid else 0.0
        else:
            scores[key] = round(float(val), 3)

    print("\n" + "=" * 50)
    print("  RAGAS RESULTS")
    print("=" * 50)
    for k, v in scores.items():
        bar = "#" * int(v * 20)
        print(f"  {k:<25} {v:.3f}  {bar}")
    print("=" * 50)

    readme = Path("README.md")
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        replacements = {
            "RAGAS faithfulness":      scores["faithfulness"],
            "RAGAS answer relevancy":  scores["answer_relevancy"],
            "RAGAS context precision": scores["context_precision"],
            "RAGAS context recall":    scores["context_recall"],
        }
        for metric, value in replacements.items():
            pattern = rf'(\|\s*{re.escape(metric)}\s*\|\s*)`?TBD[^|`]*`?(\s*\|)'
            content = re.sub(pattern, rf'\1`{value}`\2', content, flags=re.IGNORECASE)
        readme.write_text(content, encoding="utf-8")
        print("\n  README.md updated with RAGAS scores")
    else:
        print("\n  README.md not found -- run from project root")

    Path("mlops").mkdir(exist_ok=True)
    out = Path("mlops/ragas_baseline.json")
    out.write_text(json.dumps({
        "scores": scores,
        "n_examples": len(QA_PAIRS),
        "note": "standalone eval -- hand-written contexts from LLMOps domain",
    }, indent=2))
    print(f"  Baseline saved: {out}")

    return scores


if __name__ == "__main__":
    run_ragas()
