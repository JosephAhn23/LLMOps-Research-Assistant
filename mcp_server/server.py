"""
MCP (Model Context Protocol) Server for LLMOps Research Assistant.
Closes gap: MCP server experience

Exposes RAG pipeline capabilities as MCP tools so any MCP-compatible
AI agent (Claude, Cursor, etc.) can call retrieve, ingest, and evaluate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_BENCHMARK_URL = os.getenv("BENCHMARK_URL", "http://localhost:8001")

app = Server("llmops-research-assistant")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="retrieve",
            description=(
                "Retrieve relevant documents from the RAG knowledge base "
                "using semantic search + cross-encoder reranking."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of documents to return",
                    },
                    "rerank": {
                        "type": "boolean",
                        "default": True,
                        "description": "Apply cross-encoder reranking",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ingest_document",
            description=(
                "Ingest a new document into the knowledge base "
                "(chunked, embedded, indexed into FAISS)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document text content",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (source, title, date)",
                    },
                    "chunk_size": {"type": "integer", "default": 512},
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="evaluate_rag",
            description=(
                "Run RAGAS evaluation on a query-answer-context triple to "
                "measure faithfulness, relevancy, and precision."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "contexts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["question", "answer", "contexts"],
            },
        ),
        types.Tool(
            name="list_models",
            description=(
                "List available models in the SageMaker model registry "
                "with their approval status."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="benchmark_vllm",
            description=(
                "Run vLLM throughput/latency benchmark and return "
                "p50/p90/p99 latency + QPS metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "default": "meta-llama/Llama-3.1-8B-Instruct",
                    },
                    "num_prompts": {"type": "integer", "default": 100},
                    "concurrency": {"type": "integer", "default": 10},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Route MCP tool calls to the appropriate pipeline component."""
    try:
        if name == "retrieve":
            result = await _handle_retrieve(arguments)
        elif name == "ingest_document":
            result = await _handle_ingest(arguments)
        elif name == "evaluate_rag":
            result = await _handle_evaluate(arguments)
        elif name == "list_models":
            result = await _handle_list_models()
        elif name == "benchmark_vllm":
            result = await _handle_benchmark(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception:
        logger.exception("Tool %s failed", name)
        result = {"error": f"Tool {name} failed"}

    return [
        types.TextContent(type="text", text=json.dumps(result, indent=2))
    ]


async def _handle_retrieve(args: dict) -> dict:
    """Retrieve documents from distributed FAISS shards via the FastAPI gateway."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_API_BASE_URL}/retrieve",
            json={
                "query": args["query"],
                "top_k": args.get("top_k", 5),
                "rerank": args.get("rerank", True),
            },
            timeout=30.0,
        )
        return response.json()


async def _handle_ingest(args: dict) -> dict:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_API_BASE_URL}/ingest",
            json={
                "content": args["content"],
                "metadata": args.get("metadata", {}),
                "chunk_size": args.get("chunk_size", 512),
            },
            timeout=60.0,
        )
        return response.json()


async def _handle_evaluate(args: dict) -> dict:
    """Run RAGAS evaluation metrics on a RAG response."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    data = Dataset.from_dict({
        "question": [args["question"]],
        "answer": [args["answer"]],
        "contexts": [args["contexts"]],
    })
    result = evaluate(
        data, metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }


async def _handle_list_models() -> dict:
    """List registered models from SageMaker model registry."""
    import boto3

    sm = boto3.client("sagemaker")
    response = sm.list_model_packages(
        ModelPackageGroupName="llmops-research-assistant"
    )
    return {
        "models": [
            {
                "arn": pkg["ModelPackageArn"],
                "version": pkg["ModelPackageVersion"],
                "status": pkg["ModelApprovalStatus"],
                "created": str(pkg["CreationTime"]),
            }
            for pkg in response["ModelPackageSummaryList"]
        ]
    }


async def _handle_benchmark(args: dict) -> dict:
    """Trigger vLLM benchmark and return latency/throughput stats."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_BENCHMARK_URL}/benchmark",
            json=args,
            timeout=300.0,
        )
        return response.json()


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream, app.create_initialization_options()
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
