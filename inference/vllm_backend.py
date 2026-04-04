"""
vLLM inference backend - high-throughput self-hosted LLM serving.
Covers: vLLM, SGLang-compatible interface, inference systems
"""
import asyncio
import uuid
from typing import Dict, List

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a precise research assistant. Answer using ONLY
the provided context. Cite sources using [source_N] notation."""


class VLLMSynthesizer:
    """
    High-throughput LLM inference with vLLM.
    Covers: vLLM, token serving, batch inference systems
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ):
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="float16",
        )
        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
            stop=["</s>", "<|eot_id|>"],
        )

    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            context_str += f"[source_{i+1}] {chunk['source']}:\n{chunk['text']}\n\n"

        return f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Context:
{context_str}

Question: {query}
<|assistant|>"""

    def synthesize(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Single query inference."""
        prompt = self._build_prompt(query, context_chunks)
        outputs = self.llm.generate([prompt], self.sampling_params)
        answer = outputs[0].outputs[0].text.strip()

        return {
            "answer": answer,
            "sources": [c["source"] for c in context_chunks],
            "tokens_generated": len(outputs[0].outputs[0].token_ids),
            "finish_reason": outputs[0].outputs[0].finish_reason,
        }

    def batch_synthesize(self, queries: List[Dict]) -> List[Dict]:
        """
        True batch inference - vLLM processes all prompts in parallel.
        Covers: Batch inference pipelines
        """
        prompts = [
            self._build_prompt(q["query"], q["context_chunks"])
            for q in queries
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for i, output in enumerate(outputs):
            results.append(
                {
                    "query": queries[i]["query"],
                    "answer": output.outputs[0].text.strip(),
                    "sources": [c["source"] for c in queries[i]["context_chunks"]],
                    "tokens_generated": len(output.outputs[0].token_ids),
                }
            )
        return results


class AsyncVLLMSynthesizer:
    """
    Async vLLM for streaming + concurrent request handling.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def stream_synthesize(self, query: str, context_chunks: List[Dict]):
        """Streaming token generation."""
        prompt = self._build_prompt(query, context_chunks)
        request_id = str(uuid.uuid4())

        sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)

        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                yield output.outputs[0].text

    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_str = "\n\n".join(
            f"[source_{i+1}] {c['source']}:\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )
        return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nContext:\n{context_str}\n\nQuestion: {query}\n<|assistant|>"


def _run_batch_sync(vllm_synth: VLLMSynthesizer, batch: List[Dict]) -> List[Dict]:
    return vllm_synth.batch_synthesize(batch)


def run_batch_sync(batch: List[Dict]) -> List[Dict]:
    """Convenience helper for scripting contexts."""
    return _run_batch_sync(VLLMSynthesizer(), batch)


async def run_stream(query: str, context_chunks: List[Dict]) -> List[str]:
    """Collect stream outputs for async callers."""
    synth = AsyncVLLMSynthesizer()
    chunks: List[str] = []
    async for piece in synth.stream_synthesize(query, context_chunks):
        chunks.append(piece)
    return chunks


def run_stream_sync(query: str, context_chunks: List[Dict]) -> List[str]:
    return asyncio.run(run_stream(query, context_chunks))
