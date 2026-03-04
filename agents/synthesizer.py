"""
Synthesizer Agent - LLM response generation with citation tracking.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question "
    "using ONLY the provided context. Always cite your sources using [source_N] notation. "
    "If the context doesn't contain enough information, say so explicitly."
)


class SynthesizerAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        backend: str = "",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.backend = backend or os.getenv("SYNTH_BACKEND", "openai")
        self.client = None
        self.vllm_llm = None
        self.vllm_sampling = None
        self._ready = False

        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams

                vllm_model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3-8B-Instruct")
                self.vllm_llm = LLM(model=vllm_model)
                self.vllm_sampling = SamplingParams(
                    temperature=0.2, max_tokens=self.max_tokens
                )
                self._ready = True
                logger.info("Synthesizer using vLLM backend: %s", vllm_model)
                return
            except Exception as exc:
                logger.warning("vLLM unavailable, falling back to OpenAI: %s", exc)
                self.backend = "openai"

        try:
            from openai import OpenAI

            self.client = OpenAI()
            self._ready = True
        except Exception as exc:
            logger.warning("OpenAI client unavailable: %s", exc)

    def _build_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(context_chunks):
            parts.append(f"[source_{i + 1}] (from {chunk['source']}):\n{chunk['text']}")
        return "\n\n".join(parts)

    def synthesize(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_str = self._build_context(context_chunks)

        if not self._ready:
            return {
                "answer": "No LLM backend configured. Returning context-only fallback answer.",
                "sources": [c["source"] for c in context_chunks],
                "tokens_used": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        if self.backend == "vllm" and self.vllm_llm is not None:
            return self._synthesize_vllm(query, context_str, context_chunks)

        return self._synthesize_openai(query, context_str, context_chunks)

    def _synthesize_vllm(
        self, query: str, context_str: str, context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\n"
            f"Question: {query}\nAnswer with citations."
        )
        output = self.vllm_llm.generate([prompt], self.vllm_sampling)[0]
        generated = output.outputs[0] if output.outputs else None
        answer = generated.text if generated is not None else "No generation returned by vLLM."
        completion_tokens = len(getattr(generated, "token_ids", []) or [])
        prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
        return {
            "answer": answer,
            "sources": [c["source"] for c in context_chunks],
            "tokens_used": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _synthesize_openai(
        self, query: str, context_str: str, context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    timeout=float(os.getenv("OPENAI_TIMEOUT_S", "30")),
                )
                usage = response.usage
                return {
                    "answer": response.choices[0].message.content or "",
                    "sources": [c["source"] for c in context_chunks],
                    "tokens_used": usage.total_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                }
            except Exception as exc:
                last_exc = exc
                logger.warning("OpenAI attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
                time.sleep(RETRY_BACKOFF_S * attempt)
        raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} retries") from last_exc
