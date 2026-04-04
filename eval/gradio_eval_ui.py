"""
Gradio-based interactive evaluation UI for LLMOps Research Assistant.

Enables researchers and non-technical stakeholders to:
  - Query the RAG pipeline interactively
  - Compare outputs across models / retrieval backends
  - Rate responses (human preference labels for RLHF)
  - Run quantization comparisons side-by-side
  - Inspect retrieved context and reranking scores
  - Export feedback as JSONL for downstream RLHF training

Tabs:
  1. RAG Query          — single-turn query with context inspection
  2. Model Comparison   — A/B test two models on the same query
  3. Quantization Bench — Q4 vs Q8 vs F16 latency/quality comparison
  4. Human Feedback     — rate responses, export preference pairs
  5. Metrics Dashboard  — live RAGAS + DeepEval score charts

Run:
    pip install gradio
    python eval/gradio_eval_ui.py
    # Opens at http://localhost:7860

    # With custom inference backend:
    ui = LLMOpsGradioUI(inference_fn=my_rag_pipeline)
    ui.launch(share=True)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Feedback data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    """Human preference label — used for RLHF training data collection."""
    record_id: str
    query: str
    response_a: str
    response_b: str
    preferred: str          # "A" | "B" | "tie" | "both_bad"
    rating_a: int           # 1–5
    rating_b: int           # 1–5
    comment: str
    model_a: str
    model_b: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_chunks: List[str] = field(default_factory=list)

    def to_rlhf_pair(self) -> dict:
        """Convert to Anthropic HH-RLHF format for direct use with TRL DPO/RLHF trainers."""
        chosen = self.response_a if self.preferred == "A" else self.response_b
        rejected = self.response_b if self.preferred == "A" else self.response_a
        return {
            "chosen": f"Human: {self.query}\n\nAssistant: {chosen}",
            "rejected": f"Human: {self.query}\n\nAssistant: {rejected}",
            "metadata": {
                "record_id": self.record_id,
                "timestamp": self.timestamp,
                "rating_chosen": max(self.rating_a, self.rating_b),
                "rating_rejected": min(self.rating_a, self.rating_b),
            },
        }


class FeedbackStore:
    """Persist human feedback labels to JSONL for RLHF data collection."""

    def __init__(self, path: str = "data/human_feedback.jsonl"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def save(self, record: FeedbackRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")
        logger.info("Feedback saved: %s", record.record_id)

    def load_all(self) -> List[FeedbackRecord]:
        if not Path(self.path).exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                try:
                    records.append(FeedbackRecord(**json.loads(line.strip())))
                except Exception:
                    pass
        return records

    def export_rlhf_pairs(self, output_path: str = "data/rlhf_pairs.jsonl") -> int:
        """Export preference pairs in Anthropic HH-RLHF format."""
        records = self.load_all()
        pairs = [r.to_rlhf_pair() for r in records if r.preferred in ("A", "B")]
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Exported %d RLHF pairs to %s", len(pairs), output_path)
        return len(pairs)

    def stats(self) -> dict:
        records = self.load_all()
        if not records:
            return {
                "total": 0, "preferred_A": 0, "preferred_B": 0,
                "ties": 0, "both_bad": 0,
                "avg_rating_A": 0.0, "avg_rating_B": 0.0,
            }
        from collections import Counter

        pref = Counter(r.preferred for r in records)
        return {
            "total": len(records),
            "preferred_A": pref.get("A", 0),
            "preferred_B": pref.get("B", 0),
            "ties": pref.get("tie", 0),
            "both_bad": pref.get("both_bad", 0),
            "avg_rating_A": sum(r.rating_a for r in records) / len(records),
            "avg_rating_B": sum(r.rating_b for r in records) / len(records),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Mock inference (replace with real pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def _mock_inference(
    query: str, model: str = "gpt-4o-mini"
) -> Tuple[str, List[str], float]:
    """
    Returns (answer, context_chunks, latency_ms).
    Swap this out for the real RAG pipeline in production.
    """
    t0 = time.perf_counter()
    time.sleep(0.05)
    answer = (
        f"[{model}] Mock response to: '{query[:60]}'\n\n"
        "In production this is generated by the full RAG pipeline "
        "(FAISS retrieval → cross-encoder reranking → LLM synthesis)."
    )
    chunks = [
        "[Source 1] RAG combines dense retrieval with LLM generation for grounded answers.",
        "[Source 2] BM25 is a sparse retrieval model based on TF-IDF term weighting.",
        "[Source 3] Cross-encoder reranking improves precision at the cost of latency.",
    ]
    latency = (time.perf_counter() - t0) * 1000
    return answer, chunks, latency


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

class LLMOpsGradioUI:
    """
    Five-tab Gradio evaluation UI for the LLMOps Research Assistant.

    Designed for both technical researchers (context inspection, quant benchmarks)
    and non-technical stakeholders (simple query interface, preference labelling).
    """

    AVAILABLE_MODELS = [
        "gpt-4o-mini",
        "llama-3.2-1b-q4",
        "llama-3.1-8b-q4",
        "llama-3.1-8b-f16",
    ]

    QUANT_LEVELS = ["q3_k_m", "q4_k_m", "q5_k_m", "q8_0", "f16"]

    def __init__(
        self,
        inference_fn: Optional[Callable] = None,
        feedback_path: str = "data/human_feedback.jsonl",
        metrics_path: str = "mlops/baselines/",
    ):
        self.inference_fn = inference_fn or _mock_inference
        self.feedback_store = FeedbackStore(feedback_path)
        self.metrics_path = metrics_path
        self._app = None

    # ── Tab 1: RAG Query ─────────────────────────────────────────────────────

    def _query_rag(
        self, query: str, model: str, top_k: int, show_context: bool
    ) -> Tuple[str, str, str]:
        if not query.strip():
            return "Please enter a query.", "", ""
        answer, chunks, latency = self.inference_fn(query, model)
        context_text = (
            "\n\n---\n\n".join(chunks[:top_k]) if show_context else "(context hidden)"
        )
        stats = (
            f"Model: {model} | Latency: {latency:.0f} ms | Chunks retrieved: {len(chunks)}"
        )
        return answer, context_text, stats

    # ── Tab 2: Model Comparison ──────────────────────────────────────────────

    def _compare_models(
        self, query: str, model_a: str, model_b: str
    ) -> Tuple[str, str, str, str]:
        if not query.strip():
            return "Enter a query.", "", "", ""
        ans_a, _, lat_a = self.inference_fn(query, model_a)
        ans_b, _, lat_b = self.inference_fn(query, model_b)
        stats = (
            f"Model A ({model_a}): {lat_a:.0f} ms  |  Model B ({model_b}): {lat_b:.0f} ms"
        )
        return ans_a, ans_b, stats, str(time.time())

    def _save_feedback(
        self,
        query: str,
        ans_a: str,
        ans_b: str,
        model_a: str,
        model_b: str,
        preferred: str,
        rating_a: int,
        rating_b: int,
        comment: str,
        _state: str,
    ) -> str:
        record = FeedbackRecord(
            record_id=str(uuid.uuid4()),
            query=query,
            response_a=ans_a,
            response_b=ans_b,
            preferred=preferred,
            rating_a=int(rating_a),
            rating_b=int(rating_b),
            comment=comment,
            model_a=model_a,
            model_b=model_b,
        )
        self.feedback_store.save(record)
        s = self.feedback_store.stats()
        return (
            f"Saved! Total: {s['total']} | "
            f"A preferred: {s['preferred_A']} | B preferred: {s['preferred_B']} | "
            f"Ties: {s['ties']}"
        )

    # ── Tab 3: Quantization Benchmark ────────────────────────────────────────

    def _run_quant_bench(
        self, query: str, quant_levels: List[str]
    ) -> List[List[str]]:
        rows = []
        for quant in quant_levels:
            _, _, latency = self.inference_fn(query, f"llama-3.2-1b-{quant}")
            # Size estimates (MB) for a 1B model at each quant level
            size_map = {
                "q3_k_m": "~600 MB", "q4_k_m": "~750 MB",
                "q5_k_m": "~900 MB", "q8_0": "~1.3 GB", "f16": "~2.5 GB",
            }
            rows.append([quant, f"{latency:.0f} ms", "—", size_map.get(quant, "—")])
        return rows

    # ── Tab 4: Human Feedback ────────────────────────────────────────────────

    def _feedback_stats(self) -> str:
        s = self.feedback_store.stats()
        if s["total"] == 0:
            return "No feedback recorded yet."
        return (
            f"Total records: {s['total']}\n"
            f"Preferred A: {s['preferred_A']}  |  Preferred B: {s['preferred_B']}\n"
            f"Ties: {s['ties']}  |  Both bad: {s['both_bad']}\n"
            f"Avg rating A: {s['avg_rating_A']:.2f}  |  Avg rating B: {s['avg_rating_B']:.2f}"
        )

    def _export_rlhf(self) -> str:
        n = self.feedback_store.export_rlhf_pairs()
        return f"Exported {n} RLHF preference pairs to data/rlhf_pairs.jsonl"

    # ── Tab 5: Metrics Dashboard ─────────────────────────────────────────────

    def _load_metrics(self) -> List[List[str]]:
        ragas_path = Path(self.metrics_path) / "ragas_baseline.json"
        deepeval_path = Path(self.metrics_path) / "deepeval_baseline.json"
        rows: List[List[str]] = []

        for path, framework in [(ragas_path, "RAGAS"), (deepeval_path, "DeepEval")]:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for metric, vals in data.get("metrics", data).items():
                    score = vals.get("score", vals) if isinstance(vals, dict) else vals
                    try:
                        score_f = float(score)
                        rows.append([
                            framework, metric, f"{score_f:.4f}",
                            "PASS" if score_f >= 0.70 else "FAIL",
                        ])
                    except (TypeError, ValueError):
                        rows.append([framework, metric, str(score), "—"])
            else:
                rows.append([framework, "— no baseline found —", "—", "—"])

        return rows or [["—", "No metrics available", "—", "—"]]

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        try:
            import gradio as gr
        except ImportError:
            raise ImportError("gradio required: pip install gradio")

        with gr.Blocks(
            title="LLMOps Research Assistant — Eval UI",
            theme=gr.themes.Soft(),
            css=".gradio-container { max-width: 1200px !important }",
        ) as app:

            gr.Markdown(
                "# LLMOps Research Assistant — Interactive Evaluation UI\n"
                "Compare models, inspect retrieval context, collect human feedback, "
                "and monitor quality metrics."
            )

            # ── Tab 1: RAG Query ──────────────────────────────────────────
            with gr.Tab("RAG Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        q_input = gr.Textbox(
                            label="Query",
                            placeholder="e.g. What is retrieval-augmented generation?",
                            lines=3,
                        )
                        with gr.Row():
                            q_model = gr.Dropdown(
                                choices=self.AVAILABLE_MODELS,
                                value="gpt-4o-mini",
                                label="Model",
                            )
                            q_topk = gr.Slider(1, 20, value=10, step=1, label="Top-K chunks")
                        q_show_ctx = gr.Checkbox(value=True, label="Show retrieved context")
                        q_btn = gr.Button("Run Query", variant="primary")
                    with gr.Column(scale=3):
                        q_answer = gr.Textbox(label="Answer", lines=8)
                        q_stats = gr.Textbox(label="Stats", lines=1)
                        q_context = gr.Textbox(label="Retrieved Context", lines=10)

                q_btn.click(
                    fn=self._query_rag,
                    inputs=[q_input, q_model, q_topk, q_show_ctx],
                    outputs=[q_answer, q_context, q_stats],
                )

            # ── Tab 2: Model Comparison ───────────────────────────────────
            with gr.Tab("Model Comparison"):
                gr.Markdown("### A/B test two models on the same query")
                with gr.Row():
                    cmp_query = gr.Textbox(label="Query", lines=2, scale=3)
                    cmp_ma = gr.Dropdown(
                        choices=self.AVAILABLE_MODELS, value="gpt-4o-mini", label="Model A"
                    )
                    cmp_mb = gr.Dropdown(
                        choices=self.AVAILABLE_MODELS, value="llama-3.2-1b-q4", label="Model B"
                    )
                cmp_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    cmp_ans_a = gr.Textbox(label="Response A", lines=8)
                    cmp_ans_b = gr.Textbox(label="Response B", lines=8)
                cmp_stats = gr.Textbox(label="Latency", lines=1)
                cmp_state = gr.Textbox(visible=False)

                cmp_btn.click(
                    fn=self._compare_models,
                    inputs=[cmp_query, cmp_ma, cmp_mb],
                    outputs=[cmp_ans_a, cmp_ans_b, cmp_stats, cmp_state],
                )

                gr.Markdown("### Rate these responses (saved for RLHF training)")
                with gr.Row():
                    fb_pref = gr.Radio(
                        choices=["A", "B", "tie", "both_bad"],
                        value="A",
                        label="Preferred response",
                    )
                    fb_ra = gr.Slider(1, 5, value=3, step=1, label="Rating A (1–5)")
                    fb_rb = gr.Slider(1, 5, value=3, step=1, label="Rating B (1–5)")
                fb_comment = gr.Textbox(label="Comment (optional)", lines=2)
                fb_btn = gr.Button("Save Feedback", variant="secondary")
                fb_status = gr.Textbox(label="Status", lines=1)

                fb_btn.click(
                    fn=self._save_feedback,
                    inputs=[
                        cmp_query, cmp_ans_a, cmp_ans_b,
                        cmp_ma, cmp_mb,
                        fb_pref, fb_ra, fb_rb, fb_comment, cmp_state,
                    ],
                    outputs=[fb_status],
                )

            # ── Tab 3: Quantization Benchmark ─────────────────────────────
            with gr.Tab("Quantization Benchmark"):
                gr.Markdown(
                    "### Compare latency and model size across GGUF quantization levels\n"
                    "Quality score requires a live model — shows `—` in mock mode."
                )
                qb_query = gr.Textbox(
                    label="Benchmark prompt",
                    value="Explain the difference between BM25 and dense retrieval.",
                    lines=2,
                )
                qb_levels = gr.CheckboxGroup(
                    choices=self.QUANT_LEVELS,
                    value=["q4_k_m", "q8_0", "f16"],
                    label="Quantization levels",
                )
                qb_btn = gr.Button("Run Benchmark", variant="primary")
                qb_table = gr.Dataframe(
                    headers=["Quantization", "Latency", "Quality Score", "Model Size"],
                    label="Results",
                )
                qb_btn.click(
                    fn=self._run_quant_bench,
                    inputs=[qb_query, qb_levels],
                    outputs=[qb_table],
                )

            # ── Tab 4: Human Feedback ─────────────────────────────────────
            with gr.Tab("Human Feedback"):
                gr.Markdown(
                    "### Feedback dataset for RLHF training\n"
                    "Preference pairs are exported in Anthropic HH-RLHF format "
                    "and can be used directly with TRL DPO / PPO trainers."
                )
                with gr.Row():
                    hf_stats_btn = gr.Button("Refresh Stats")
                    hf_export_btn = gr.Button("Export RLHF Pairs", variant="primary")
                hf_stats_out = gr.Textbox(label="Feedback Statistics", lines=6)
                hf_export_out = gr.Textbox(label="Export Status", lines=1)

                hf_stats_btn.click(fn=self._feedback_stats, outputs=[hf_stats_out])
                hf_export_btn.click(fn=self._export_rlhf, outputs=[hf_export_out])

                gr.Markdown(
                    "**Output format** (`data/rlhf_pairs.jsonl`):\n"
                    "```json\n"
                    '{"chosen": "Human: <query>\\n\\nAssistant: <preferred>",\n'
                    ' "rejected": "Human: <query>\\n\\nAssistant: <non-preferred>",\n'
                    ' "metadata": {"record_id": "...", "rating_chosen": 4, ...}}\n'
                    "```"
                )

            # ── Tab 5: Metrics Dashboard ──────────────────────────────────
            with gr.Tab("Metrics Dashboard"):
                gr.Markdown(
                    "### Live RAGAS + DeepEval scores\n"
                    "Reads from `mlops/baselines/ragas_baseline.json` and "
                    "`mlops/baselines/deepeval_baseline.json`."
                )
                md_refresh_btn = gr.Button("Refresh Metrics")
                md_table = gr.Dataframe(
                    headers=["Framework", "Metric", "Score", "Status"],
                    label="Evaluation Scores",
                )
                md_refresh_btn.click(fn=self._load_metrics, outputs=[md_table])

        self._app = app
        return app

    def launch(
        self,
        host: str = "0.0.0.0",
        port: int = 7860,
        share: bool = False,
        auth: Optional[Tuple[str, str]] = None,
    ) -> None:
        app = self.build()
        logger.info("Launching Gradio UI at http://%s:%d", host, port)
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
            auth=auth,
            show_error=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMOps Gradio evaluation UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--feedback-path", default="data/human_feedback.jsonl")
    parser.add_argument("--metrics-path", default="mlops/baselines/")
    args = parser.parse_args()

    ui = LLMOpsGradioUI(
        feedback_path=args.feedback_path,
        metrics_path=args.metrics_path,
    )
    ui.launch(host=args.host, port=args.port, share=args.share)
