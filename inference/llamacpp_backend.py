"""
llama.cpp Local Inference Backend + SLM Evaluation Suite
=========================================================
Local inference via llama.cpp — zero cloud dependency, runs on CPU or GPU.

Features:
  - llama-cpp-python backend (bindings to llama.cpp)
  - GGUF model loading (Q4_K_M, Q5_K_M, Q8_0, F16)
  - SLM (Small Language Model) evaluation suite:
      - Task-based testing (QA, summarisation, classification, reasoning)
      - Long-context behaviour testing (needle-in-haystack)
      - Contamination detection via n-gram overlap
      - Conversation dynamics (multi-turn coherence)
  - Quantization comparison: Q4 vs Q5 vs Q8 vs F16
  - GGUFModelManager: download and cache GGUF models from HuggingFace Hub

Usage:
    runner = LlamaCppRunner.from_gguf("models/llama-3.2-1b-instruct-q4_k_m.gguf")
    result = runner.generate("Explain RAG in one sentence.")
    print(result)

    evaluator = SLMEvaluator(runner)
    report = evaluator.run_full_suite()
    print(report.summary())

Run locally (CPU, no GPU needed):
    python inference/llamacpp_backend.py --eval --model <path-to.gguf>
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LlamaCppConfig:
    model_path: str = "models/model.gguf"
    n_ctx: int = 4096           # context window
    n_threads: int = 8          # CPU threads (0 = auto)
    n_gpu_layers: int = 0       # layers to offload to GPU (0 = CPU only; -1 = all)
    n_batch: int = 512          # prompt processing batch size
    temperature: float = 0.0    # 0 = deterministic
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Generation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_second: float
    model_path: str
    finish_reason: str          # "stop" | "length" | "error"

    def __str__(self) -> str:
        return (
            f"[{self.finish_reason}] {self.text[:200]}\n"
            f"  tokens: {self.prompt_tokens}+{self.completion_tokens}={self.total_tokens} | "
            f"  TTFT: {self.time_to_first_token_ms:.1f}ms | "
            f"  total: {self.total_latency_ms:.1f}ms | "
            f"  {self.tokens_per_second:.1f} tok/s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# GGUF Model Manager
# ──────────────────────────────────────────────────────────────────────────────

class GGUFModelManager:
    """
    Downloads and caches GGUF model files from HuggingFace Hub.
    GGUF is the standard format for llama.cpp quantised models.
    """

    def __init__(self, cache_dir: str = "~/.cache/llama_cpp"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, repo_id: str, filename: str) -> str:
        """
        Return local path to a GGUF model, downloading from HF Hub if needed.

        Args:
            repo_id:  HuggingFace repo, e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF"
            filename: GGUF filename, e.g. "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        """
        local_path = self.cache_dir / filename
        if local_path.exists():
            logger.info("GGUF model cached: %s", local_path)
            return str(local_path)

        logger.info("Downloading GGUF model: %s / %s", repo_id, filename)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.cache_dir),
        )
        logger.info("Downloaded to: %s", path)
        return path

    def list_cached(self) -> List[str]:
        return [str(p) for p in self.cache_dir.glob("*.gguf")]

    @staticmethod
    def recommended_quant(vram_gb: float) -> str:
        """Recommend a quantisation level based on available VRAM / RAM."""
        if vram_gb >= 16:
            return "Q8_0"
        elif vram_gb >= 8:
            return "Q6_K"
        elif vram_gb >= 6:
            return "Q5_K_M"
        elif vram_gb >= 4:
            return "Q4_K_M"
        else:
            return "Q3_K_M"


# ──────────────────────────────────────────────────────────────────────────────
# llama.cpp runner
# ──────────────────────────────────────────────────────────────────────────────

class LlamaCppRunner:
    """
    Wraps llama-cpp-python for local inference.
    Supports GGUF models at all quantisation levels.

    Drop-in alternative to the vLLM backend for:
      - CPU-only environments (developer laptops, CI)
      - Air-gapped / offline deployments
      - Apple Silicon (Metal via n_gpu_layers=-1)
      - Edge devices (Jetson Nano, Raspberry Pi)
    """

    def __init__(self, cfg: LlamaCppConfig):
        self.cfg = cfg
        self._model = None
        self._load_time_ms: float = 0.0

    @classmethod
    def from_gguf(
        cls,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
    ) -> "LlamaCppRunner":
        cfg = LlamaCppConfig(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        return cls(cfg)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        n_gpu_layers: int = 0,
    ) -> "LlamaCppRunner":
        """Download a GGUF model from HF Hub and return a runner."""
        manager = GGUFModelManager()
        path = manager.get_model_path(repo_id, filename)
        return cls.from_gguf(path, n_gpu_layers=n_gpu_layers)

    def _load_model(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "CPU:    pip install llama-cpp-python\n"
                "CUDA:   CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python\n"
                "Metal:  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            )

        n_threads = self.cfg.n_threads or os.cpu_count()
        t0 = time.perf_counter()
        self._model = Llama(
            model_path=self.cfg.model_path,
            n_ctx=self.cfg.n_ctx,
            n_threads=n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            n_batch=self.cfg.n_batch,
            use_mmap=self.cfg.use_mmap,
            use_mlock=self.cfg.use_mlock,
            verbose=self.cfg.verbose,
        )
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "llama.cpp model loaded in %.1fms (n_gpu_layers=%d): %s",
            self._load_time_ms, self.cfg.n_gpu_layers, self.cfg.model_path,
        )

    # ── Text completion ──────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> GenerationResult:
        if self._model is None:
            self._load_model()

        max_tokens = max_tokens or self.cfg.max_tokens
        t0 = time.perf_counter()

        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            repeat_penalty=self.cfg.repeat_penalty,
            stop=["</s>", "<|end|>", "<|eot_id|>"],
            echo=False,
        )

        total_ms = (time.perf_counter() - t0) * 1000
        usage = output.get("usage", {})
        text = output["choices"][0]["text"]
        completion_tokens = usage.get("completion_tokens", len(text.split()))

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            time_to_first_token_ms=total_ms * 0.1,   # estimated; llama-cpp doesn't expose TTFT directly
            total_latency_ms=total_ms,
            tokens_per_second=completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0,
            model_path=self.cfg.model_path,
            finish_reason=output["choices"][0].get("finish_reason", "stop"),
        )

    def stream(self, prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Stream tokens as they are generated."""
        if self._model is None:
            self._load_model()
        for chunk in self._model(
            prompt,
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=True,
        ):
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token

    # ── Chat completion ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> GenerationResult | Iterator[str]:
        """
        OpenAI-compatible chat interface.
        messages: [{"role": "system"|"user"|"assistant", "content": str}]
        """
        if self._model is None:
            self._load_model()

        if stream:
            return self._stream_chat(messages, max_tokens)

        t0 = time.perf_counter()
        output = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        usage = output.get("usage", {})
        text = output["choices"][0]["message"]["content"]
        completion_tokens = usage.get("completion_tokens", len(text.split()))

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            time_to_first_token_ms=total_ms * 0.1,
            total_latency_ms=total_ms,
            tokens_per_second=completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0,
            model_path=self.cfg.model_path,
            finish_reason=output["choices"][0].get("finish_reason", "stop"),
        )

    def _stream_chat(
        self, messages: List[Dict], max_tokens: Optional[int]
    ) -> Iterator[str]:
        for chunk in self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    # ── RAG synthesis (matches SynthesizerAgent interface) ───────────────────

    def synthesize(
        self,
        query: str,
        documents: List[Dict],
        stream: bool = False,
    ) -> GenerationResult | Iterator[str]:
        """
        RAG synthesis — formats retrieved documents as context and generates
        a grounded answer. Matches the SynthesizerAgent.synthesize() interface.
        """
        context = "\n\n".join(
            f"[source_{i}] {doc.get('text', doc.get('content', ''))[:400]}"
            for i, doc in enumerate(documents[:5], start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. Answer using ONLY the "
                    "provided context. Cite sources with [source_N] notation."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return self.chat(messages, stream=stream)

    # ── Tokenisation utilities ───────────────────────────────────────────────

    def tokenize(self, text: str) -> List[int]:
        if self._model is None:
            self._load_model()
        return self._model.tokenize(text.encode("utf-8"))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def fits_in_context(self, text: str, reserve: int = 512) -> bool:
        return self.count_tokens(text) <= (self.cfg.n_ctx - reserve)

    # ── Benchmarking ─────────────────────────────────────────────────────────

    def benchmark(self, prompts: List[str], n_runs: int = 50) -> Dict:
        """Measure TTFT, total latency, and tokens/s across prompts."""
        import numpy as np

        results = [self.generate(p) for p in prompts[:n_runs]]
        latencies = [r.total_latency_ms for r in results]
        tps = [r.tokens_per_second for r in results]

        return {
            "model": self.cfg.model_path,
            "n_runs": len(results),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p90_ms": float(np.percentile(latencies, 90)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "tokens_per_second_mean": float(np.mean(tps)),
            "tokens_per_second_p50": float(np.percentile(tps, 50)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# SLM evaluation tasks
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalTask:
    task_id: str
    category: str       # "qa" | "summarisation" | "classification" | "reasoning" | "long_context"
    prompt: str
    expected: Optional[str] = None
    expected_labels: Optional[List[str]] = None
    context_len: int = 0


@dataclass
class EvalResult:
    task_id: str
    category: str
    prompt_preview: str
    response: str
    expected: Optional[str]
    score: float            # 0–1
    passed: bool
    latency_ms: float
    tokens_per_second: float
    notes: str = ""


@dataclass
class SLMEvalReport:
    model_path: str
    total_tasks: int
    results: List[EvalResult]
    category_scores: Dict[str, float]
    overall_score: float
    avg_latency_ms: float
    avg_tokens_per_second: float
    contamination_flags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        passed = sum(r.passed for r in self.results)
        lines = [
            f"SLM Evaluation Report — {self.model_path}",
            f"Overall: {self.overall_score:.3f} ({passed}/{self.total_tasks} passed)",
            "",
            "Category scores:",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat:20s}: {score:.3f}")
        lines += [
            "",
            f"Avg latency:    {self.avg_latency_ms:.1f}ms",
            f"Avg tok/s:      {self.avg_tokens_per_second:.1f}",
        ]
        if self.contamination_flags:
            lines.append(f"\nContamination warnings: {len(self.contamination_flags)}")
            for flag in self.contamination_flags[:3]:
                lines.append(f"  - {flag}")
        return "\n".join(lines)

    def log_to_mlflow(self) -> None:
        try:
            import mlflow
            with mlflow.start_run(run_name="slm-eval"):
                mlflow.log_param("model", self.model_path)
                mlflow.log_metric("overall_score", self.overall_score)
                mlflow.log_metric("avg_latency_ms", self.avg_latency_ms)
                mlflow.log_metric("avg_tokens_per_second", self.avg_tokens_per_second)
                for cat, score in self.category_scores.items():
                    mlflow.log_metric(f"score_{cat}", score)
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# SLM Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class SLMEvaluator:
    """
    Comprehensive Small Language Model evaluation suite.

    Tests:
      1. Task-based:        QA accuracy, summarisation quality, classification F1
      2. Reasoning:         chain-of-thought, multi-step math
      3. Long-context:      needle-in-haystack retrieval at varying depths
      4. Conversation:      multi-turn coherence and context retention
      5. Contamination:     n-gram overlap with known benchmark samples
    """

    def __init__(self, runner: LlamaCppRunner, pass_threshold: float = 0.5):
        self.runner = runner
        self.pass_threshold = pass_threshold

    # ── Scoring helpers ──────────────────────────────────────────────────────

    def _exact_match(self, response: str, expected: str) -> float:
        return 1.0 if expected.lower().strip() in response.lower() else 0.0

    def _token_overlap_f1(self, response: str, expected: str) -> float:
        """Token-level F1 — standard for open-domain QA evaluation."""
        def tokens(s):
            return set(re.findall(r"\w+", s.lower()))

        pred = tokens(response)
        gold = tokens(expected)
        if not pred or not gold:
            return 0.0
        common = pred & gold
        if not common:
            return 0.0
        precision = len(common) / len(pred)
        recall = len(common) / len(gold)
        return 2 * precision * recall / (precision + recall)

    def _classification_accuracy(self, response: str, expected_labels: List[str]) -> float:
        resp_lower = response.lower()
        return 1.0 if any(label.lower() in resp_lower for label in expected_labels) else 0.0

    def _reasoning_score(self, response: str, expected: str) -> float:
        """Check if the final answer is correct, regardless of chain-of-thought."""
        answer_patterns = [
            r"answer[:\s]+(.+?)[\n\.]",
            r"therefore[,:\s]+(.+?)[\n\.]",
            r"=\s*(\d+[\.\d]*)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response.lower())
            if match and expected.lower().strip() in match.group(1).lower():
                return 1.0
        return self._token_overlap_f1(response, expected)

    # ── Needle-in-haystack (long-context) ────────────────────────────────────

    def build_needle_in_haystack(
        self,
        needle: str,
        haystack_words: int = 2000,
        depth_pct: float = 50.0,
    ) -> tuple:
        """
        Hide a specific fact inside filler text at depth_pct% of the document.
        Returns (prompt, expected_answer).
        """
        filler = (
            "The study of machine learning encompasses supervised learning, "
            "unsupervised learning, and reinforcement learning. "
            "Neural networks have shown remarkable performance on image recognition "
            "and natural language processing. Transformers, introduced in 2017, "
            "revolutionised sequence modelling through the self-attention mechanism. "
        ) * (haystack_words // 40 + 1)

        words = filler.split()
        insert_pos = int(len(words) * depth_pct / 100)
        words.insert(insert_pos, needle)
        context = " ".join(words[:haystack_words])
        prompt = (
            f"{context}\n\n"
            "Question: What specific fact about the needle was mentioned in the text above?\n"
            "Answer:"
        )
        return prompt, needle

    def evaluate_long_context(
        self,
        depths: Optional[List[float]] = None,
        context_lengths: Optional[List[int]] = None,
    ) -> List[EvalResult]:
        depths = depths or [10.0, 25.0, 50.0, 75.0, 90.0]
        context_lengths = context_lengths or [1000, 2000, 4000]
        needle = "The secret code for the vault is ALPHA-7734-ZETA."
        expected = "ALPHA-7734-ZETA"
        results = []

        for ctx_len in context_lengths:
            for depth in depths:
                if not self.runner.fits_in_context(f"{'x ' * ctx_len}", reserve=200):
                    logger.debug("Skipping ctx_len=%d (exceeds model context)", ctx_len)
                    continue
                prompt, _ = self.build_needle_in_haystack(needle, ctx_len, depth)
                result = self.runner.generate(prompt, max_tokens=100)
                score = self._exact_match(result.text, expected)
                results.append(EvalResult(
                    task_id=f"needle_ctx{ctx_len}_depth{int(depth)}",
                    category="long_context",
                    prompt_preview=f"[{ctx_len}w, depth={depth}%] needle-in-haystack",
                    response=result.text,
                    expected=expected,
                    score=score,
                    passed=score >= self.pass_threshold,
                    latency_ms=result.total_latency_ms,
                    tokens_per_second=result.tokens_per_second,
                    notes=f"ctx_len={ctx_len}, depth={depth}%",
                ))

        return results

    # ── Conversation dynamics ─────────────────────────────────────────────────

    def evaluate_conversation_dynamics(self) -> List[EvalResult]:
        """
        Test multi-turn coherence: does the model retain context across turns?
        """
        conversations = [
            {
                "turns": [
                    {"role": "user", "content": "My name is Joseph. Remember that."},
                    {"role": "assistant", "content": "Hello Joseph! I'll remember your name."},
                    {"role": "user", "content": "What is my name?"},
                ],
                "expected": "Joseph",
                "task_id": "conv_name_recall",
            },
            {
                "turns": [
                    {"role": "user",
                     "content": "We're discussing Python programming. What language are we discussing?"},
                ],
                "expected": "Python",
                "task_id": "conv_topic_recall",
            },
            {
                "turns": [
                    {"role": "user", "content": "I prefer concise answers. Keep responses under 2 sentences."},
                    {"role": "assistant", "content": "Understood, I'll keep responses brief."},
                    {"role": "user", "content": "What is a transformer?"},
                ],
                "expected": "attention",
                "task_id": "conv_instruction_follow",
            },
        ]

        results = []
        for conv in conversations:
            result = self.runner.chat(conv["turns"])
            score = self._exact_match(result.text, conv["expected"])
            results.append(EvalResult(
                task_id=conv["task_id"],
                category="conversation",
                prompt_preview=conv["turns"][-1]["content"],
                response=result.text,
                expected=conv["expected"],
                score=score,
                passed=score >= self.pass_threshold,
                latency_ms=result.total_latency_ms,
                tokens_per_second=result.tokens_per_second,
            ))
        return results

    # ── Contamination detection ───────────────────────────────────────────────

    def detect_contamination(
        self,
        benchmark_samples: List[str],
        n_gram: int = 8,
        overlap_threshold: float = 0.4,
    ) -> List[str]:
        """
        Detect potential training contamination by measuring n-gram overlap
        between model outputs and known benchmark samples.

        High overlap suggests the model may have memorised benchmark data,
        making evaluation scores unreliable.
        """
        flags = []
        for sample in benchmark_samples:
            response = self.runner.generate(sample[:200], max_tokens=100)
            sample_ngrams = self._get_ngrams(sample, n_gram)
            response_ngrams = self._get_ngrams(response.text, n_gram)
            if not sample_ngrams:
                continue
            overlap = len(sample_ngrams & response_ngrams) / len(sample_ngrams)
            if overlap > overlap_threshold:
                flags.append(
                    f"High n-gram overlap ({overlap:.1%}) for: '{sample[:80]}...'"
                )
        return flags

    def _get_ngrams(self, text: str, n: int) -> set:
        tokens = text.lower().split()
        return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    # ── Standard task suite ───────────────────────────────────────────────────

    def build_standard_tasks(self) -> List[EvalTask]:
        return [
            # QA
            EvalTask("qa_rag", "qa",
                     "What does RAG stand for in the context of AI?",
                     "Retrieval-Augmented Generation"),
            EvalTask("qa_transformer", "qa",
                     "Who introduced the Transformer architecture in the 'Attention is All You Need' paper?",
                     "Vaswani"),
            EvalTask("qa_faiss", "qa",
                     "What does FAISS stand for?",
                     "Facebook AI Similarity Search"),
            # Summarisation
            EvalTask("sum_abstract", "summarisation",
                     "Summarise in one sentence: Transformers use self-attention to model "
                     "relationships between all tokens in a sequence simultaneously, enabling "
                     "parallelism and long-range dependency capture.",
                     "Transformers use self-attention for parallel sequence modelling."),
            # Classification
            EvalTask("cls_sentiment_pos", "classification",
                     "Classify as positive or negative: 'The model performed exceptionally well.'",
                     "positive", expected_labels=["positive"]),
            EvalTask("cls_sentiment_neg", "classification",
                     "Classify as positive or negative: 'The results were disappointing and below expectations.'",
                     "negative", expected_labels=["negative"]),
            EvalTask("cls_intent", "classification",
                     "Classify the intent: 'What is the weather today?'",
                     "question", expected_labels=["question", "informational"]),
            # Reasoning
            EvalTask("reason_math", "reasoning",
                     "If a model processes 1000 tokens/s and a prompt is 500 tokens, "
                     "how many seconds to process? Show your reasoning.",
                     "0.5"),
            EvalTask("reason_logic", "reasoning",
                     "All LLMs use attention. GPT-4 is an LLM. Does GPT-4 use attention? Answer yes or no.",
                     "yes"),
            EvalTask("reason_chain", "reasoning",
                     "A rectangle has length 8 and width 5. What is its area? Think step by step.",
                     "40"),
        ]

    def run_task(self, task: EvalTask) -> EvalResult:
        result = self.runner.generate(task.prompt)

        if task.category == "classification" and task.expected_labels:
            score = self._classification_accuracy(result.text, task.expected_labels)
        elif task.category == "reasoning" and task.expected:
            score = self._reasoning_score(result.text, task.expected)
        elif task.expected:
            score = self._token_overlap_f1(result.text, task.expected)
        else:
            score = 1.0

        return EvalResult(
            task_id=task.task_id,
            category=task.category,
            prompt_preview=task.prompt[:80],
            response=result.text,
            expected=task.expected,
            score=score,
            passed=score >= self.pass_threshold,
            latency_ms=result.total_latency_ms,
            tokens_per_second=result.tokens_per_second,
        )

    def run_full_suite(
        self,
        include_long_context: bool = True,
        include_conversation: bool = True,
        contamination_samples: Optional[List[str]] = None,
        log_mlflow: bool = False,
    ) -> SLMEvalReport:
        import numpy as np

        all_results: List[EvalResult] = []

        tasks = self.build_standard_tasks()
        logger.info("Running %d standard eval tasks…", len(tasks))
        for task in tasks:
            all_results.append(self.run_task(task))

        if include_long_context:
            logger.info("Running long-context (needle-in-haystack) tests…")
            all_results.extend(self.evaluate_long_context())

        if include_conversation:
            logger.info("Running conversation dynamics tests…")
            all_results.extend(self.evaluate_conversation_dynamics())

        contamination_flags: List[str] = []
        if contamination_samples:
            logger.info("Running contamination detection…")
            contamination_flags = self.detect_contamination(contamination_samples)

        category_scores: Dict[str, float] = {}
        for cat in set(r.category for r in all_results):
            cat_results = [r for r in all_results if r.category == cat]
            category_scores[cat] = float(np.mean([r.score for r in cat_results]))

        report = SLMEvalReport(
            model_path=self.runner.cfg.model_path,
            total_tasks=len(all_results),
            results=all_results,
            category_scores=category_scores,
            overall_score=float(np.mean([r.score for r in all_results])),
            avg_latency_ms=float(np.mean([r.latency_ms for r in all_results])),
            avg_tokens_per_second=float(np.mean([r.tokens_per_second for r in all_results])),
            contamination_flags=contamination_flags,
        )

        if log_mlflow:
            report.log_to_mlflow()

        return report


# ──────────────────────────────────────────────────────────────────────────────
# Quantization comparison
# ──────────────────────────────────────────────────────────────────────────────

class QuantizationComparison:
    """
    Compare quality vs speed across GGUF quantization levels.
    Typical quality order: F16 > Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q3_K_M
    """

    QUANT_LEVELS = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]

    def __init__(self, model_dir: str, base_model_name: str):
        self.model_dir = model_dir
        self.base_model_name = base_model_name

    def _model_path(self, quant: str) -> str:
        return f"{self.model_dir}/{self.base_model_name}-{quant}.gguf"

    def compare(
        self,
        test_prompts: List[str],
        quant_levels: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        quants = quant_levels or self.QUANT_LEVELS
        results: Dict[str, Dict] = {}

        for quant in quants:
            path = self._model_path(quant)
            if not os.path.exists(path):
                logger.warning("Model not found, skipping: %s", path)
                continue
            logger.info("Benchmarking quantization level: %s", quant)
            runner = LlamaCppRunner.from_gguf(path)
            bench = runner.benchmark(test_prompts)
            evaluator = SLMEvaluator(runner)
            tasks = evaluator.build_standard_tasks()[:5]
            task_results = [evaluator.run_task(t) for t in tasks]
            avg_score = sum(r.score for r in task_results) / max(len(task_results), 1)

            results[quant] = {
                **bench,
                "avg_quality_score": avg_score,
                "model_size_mb": os.path.getsize(path) / (1024 ** 2),
            }

        return results

    def print_comparison(self, results: Dict[str, Dict]) -> None:
        print("\n" + "=" * 80)
        print(f"{'Quant':10s} | {'Size MB':>8s} | {'p50 ms':>8s} | {'tok/s':>8s} | {'Quality':>8s}")
        print("=" * 80)
        for quant, r in results.items():
            print(
                f"{quant:10s} | {r.get('model_size_mb', 0):8.0f} | "
                f"{r.get('latency_p50_ms', 0):8.1f} | "
                f"{r.get('tokens_per_second_mean', 0):8.1f} | "
                f"{r.get('avg_quality_score', 0):8.3f}"
            )
        print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="llama.cpp inference + SLM evaluation suite")
    parser.add_argument("--model", default="models/llama-3.2-1b-instruct-q4_k_m.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="GPU layers to offload (0=CPU, -1=all)")
    parser.add_argument("--eval", action="store_true",
                        help="Run full SLM evaluation suite")
    parser.add_argument("--no-long-context", action="store_true",
                        help="Skip needle-in-haystack tests (faster)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log eval results to MLflow")
    parser.add_argument("--prompt", default="Explain retrieval-augmented generation in one sentence.",
                        help="Single prompt to generate (when --eval not set)")
    args = parser.parse_args()

    runner = LlamaCppRunner.from_gguf(args.model, n_gpu_layers=args.n_gpu_layers)

    if args.eval:
        evaluator = SLMEvaluator(runner)
        report = evaluator.run_full_suite(
            include_long_context=not args.no_long_context,
            log_mlflow=args.mlflow,
        )
        print(report.summary())
    else:
        result = runner.generate(args.prompt)
        print(result)
