"""
Transparent Research UI — Gradio chat with glass-box panels.

Shows Chain-of-Thought scaffold, consensus status, retrieved sources (RAGAS-style
trust proxy), and sandbox placeholder / WebSocket sandbox hints.

Modes
-----
1. **Transparent (REST)** — ``POST /retrieve`` + ``POST /query`` for full pipeline
   metadata (truth committee, adversarial consensus, attribution, quality alerts).
2. **Stream (WebSocket)** — ``/ws/query`` for token streaming; server emits
   ``thinking``, ``sources``, ``consensus``, ``sandbox`` before tokens.

Run (API must be up: ``uvicorn api.main:app --reload``)::

    pip install 'gradio>=4.36' websockets
    python eval/research_ui.py

    # Or:
    llmops-research-ui

Env (optional):
    RESEARCH_UI_API_BASE — default http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - optional UI extra
    raise SystemExit(
        "Gradio is required: pip install 'gradio>=4.36' (see project optional extra [ui])."
    ) from exc


def _api_base() -> str:
    return os.getenv("RESEARCH_UI_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _headers(api_key: str) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if (api_key or "").strip():
        h["X-API-Key"] = api_key.strip()
    return h


def _post_json(url: str, payload: dict, headers: Dict[str, str]) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _trust_proxy(chunk: Dict[str, Any]) -> float:
    """Heuristic 0–1 score (proxy for RAGAS faithfulness context relevance)."""
    for key in ("rerank_score", "retrieval_score", "score"):
        v = chunk.get(key)
        if v is not None:
            try:
                return min(0.99, max(0.05, float(v)))
            except (TypeError, ValueError):
                pass
    return 0.55


def _format_sources_markdown(chunks: List[Dict[str, Any]]) -> str:
    lines = [
        "| # | Trust (proxy) | Source | doc_id | ingested_at | Preview |",
        "|---:|---:|---|---|---|---|",
    ]
    for i, ch in enumerate(chunks):
        prev = (ch.get("text") or ch.get("text_preview") or "")[:320].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i + 1} | {_trust_proxy(ch):.2f} | {ch.get('source', '')} | "
            f"{ch.get('doc_id', '—')} | {ch.get('ingested_at', '—')} | {prev} |"
        )
    lines.append("\n*Trust scores are retrieval/rerank proxies unless RAGAS batch metrics are wired in.*")
    return "\n".join(lines)


def _thinking_from_chunks(query: str, chunks: List[Dict[str, Any]]) -> str:
    from context_engineering.chain_of_thought import ChainOfThoughtBuilder

    context_str = ""
    for i, chunk in enumerate(chunks):
        src = chunk.get("source", f"doc_{i + 1}")
        context_str += f"[source_{i + 1}] {src}:\n{chunk.get('text', '')}\n\n"
    try:
        return ChainOfThoughtBuilder().build(query, context_str, strategy="scratchpad")
    except Exception as exc:
        return f"[CoT fallback] {exc}\n\nQuery:\n{query[:2000]}\n\nContext chars: {len(context_str)}"


def _consensus_rest(data: Dict[str, Any]) -> Tuple[str, str]:
    """Returns (markdown_badge, detail_json)."""
    detail: Dict[str, Any] = {"raw_keys": sorted(data.keys())}

    if data.get("behavioral_blocked"):
        detail["behavioral_reasons"] = data.get("behavioral_reasons", [])
        return (
            "### 🔴 **Red — blocked**\nBehavioral / gateway constitutional gate.",
            json.dumps(detail, indent=2)[:8000],
        )

    adv = data.get("adversarial_consensus")
    if isinstance(adv, dict) and adv.get("hitl_recommended"):
        detail["adversarial"] = adv
        return (
            "### 🔴 **Red — adversarial HITL**\nNumeric or skeptic conflict after hard resets.",
            json.dumps(detail, indent=2)[:8000],
        )

    if data.get("consensus_hitl"):
        detail["consensus_discrepancy"] = data.get("consensus_discrepancy")
        return (
            "### 🔴 **Red — truth committee halted**\nIndependent models disagreed.",
            json.dumps(detail, indent=2)[:8000],
        )

    if data.get("quality_alert"):
        detail["quality_alert"] = data["quality_alert"]
        detail["effective_faithfulness"] = data.get("effective_faithfulness")
        return (
            f"### 🟠 **Amber — quality**\n{data['quality_alert']}",
            json.dumps(detail, indent=2)[:8000],
        )

    tc = data.get("truth_committee")
    if isinstance(tc, dict) and tc.get("is_consensus_reached") is False:
        detail["truth_committee"] = tc
        return (
            "### 🟠 **Amber — committee notes**\nConsensus not fully reached; see JSON.",
            json.dumps(detail, indent=2)[:8000],
        )

    detail["consensus_score"] = data.get("consensus_score")
    detail["grounding_confidence"] = data.get("grounding_confidence")
    return (
        "### 🟢 **Green — no halt flags**\nPrimary pipeline completed without HITL halts "
        "(still verify sources and numeric claims).",
        json.dumps(detail, indent=2)[:8000],
    )


def run_transparent_rest(
    query: str,
    api_base: str,
    api_key: str,
    session_id: str,
) -> Tuple[str, str, str, str, str, str]:
    """REST glass-box path."""
    query = (query or "").strip()
    if not query:
        return "", "", "", "", "", "Enter a question."

    base = (api_base or _api_base()).rstrip("/")
    hdrs = _headers(api_key)

    try:
        retrieved = _post_json(
            f"{base}/retrieve",
            {"query": query, "top_k": 8, "rerank": True},
            hdrs,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return "", "", "", "", "", f"Retrieve HTTP {e.code}: {body[:1500]}"
    except Exception as exc:
        return "", "", "", "", "", f"Retrieve error: {exc}"

    chunks = retrieved.get("results") or []
    thinking = _thinking_from_chunks(query, chunks)
    sources_md = _format_sources_markdown(chunks)

    payload: Dict[str, Any] = {"query": query}
    if (session_id or "").strip():
        payload["session_id"] = session_id.strip()

    try:
        result = _post_json(f"{base}/query", payload, hdrs)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return thinking, sources_md, "", "", "", f"Query HTTP {e.code}: {body[:1500]}"
    except Exception as exc:
        return thinking, sources_md, "", "", "", f"Query error: {exc}"

    answer = result.get("answer", "")
    badge, detail = _consensus_rest(result)
    sandbox_note = (
        "[sandbox] RAG /query path does not execute Docker code. "
        "Use agent tooling (SelfHealingLoop) and paste logs below, or extend the API to stream executor output.\n"
    )
    meta_bits = []
    if result.get("speculative_sentence_count") is not None:
        meta_bits.append(f"speculative_sentence_count={result['speculative_sentence_count']}")
    if result.get("grounding_confidence") is not None:
        meta_bits.append(f"grounding_confidence={result['grounding_confidence']:.3f}")
    prov = result.get("answer_with_provenance")
    if prov:
        answer = f"{answer}\n\n---\n**Traceable report (paragraph tags)**\n\n{prov}"
    return (
        thinking,
        sources_md,
        badge,
        answer,
        sandbox_note + "\n".join(meta_bits),
        detail,
    )


async def _ws_collect(base_http: str, query: str, api_key: str) -> Tuple[str, str, str, str, str, str]:
    try:
        import websockets
    except ImportError:
        return (
            "",
            "",
            "",
            "",
            "",
            json.dumps({"mode": "websocket", "error": "pip install websockets for stream mode."}),
        )

    ws_root = base_http.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    uri = f"{ws_root}/ws/query"
    extra_headers = [("X-API-Key", api_key.strip())] if (api_key or "").strip() else []

    thinking_txt = ""
    sources_md = ""
    consensus_md = ""
    sandbox_log = ""
    answer = ""
    err = ""

    connect_kw: Dict[str, Any] = {"max_size": 16_000_000}
    if extra_headers:
        # websockets >=11 prefers additional_headers; older uses extra_headers
        connect_kw["additional_headers"] = extra_headers

    try:
        ws_ctx = websockets.connect(uri, **connect_kw)
    except TypeError:
        connect_kw.pop("additional_headers", None)
        if extra_headers:
            connect_kw["extra_headers"] = extra_headers
        ws_ctx = websockets.connect(uri, **connect_kw)

    try:
        async with ws_ctx as ws:
            await ws.send(json.dumps({"query": query}))
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "thinking":
                    thinking_txt = msg.get("content", "")
                elif mtype == "sources":
                    sources_md = _format_sources_markdown(msg.get("chunks") or [])
                elif mtype == "consensus":
                    st = msg.get("status", "")
                    consensus_md = f"### 🟠 **{st}**\n{msg.get('detail', '')}"
                elif mtype == "sandbox":
                    sandbox_log = msg.get("log", "")
                elif mtype == "token":
                    answer += msg.get("content", "")
                elif mtype == "done":
                    answer = msg.get("full_response", answer)
                    break
                elif mtype == "error":
                    err = msg.get("message", "error")
                    break
                elif mtype == "status":
                    sandbox_log += f"\n[status] {msg.get('message', '')}"
    except Exception as exc:
        err = str(exc)

    if err:
        detail = json.dumps({"mode": "websocket", "error": err}, indent=2)
    else:
        detail = json.dumps({"mode": "websocket", "ok": True}, indent=2)
    return thinking_txt, sources_md, consensus_md, answer, sandbox_log, detail


def run_websocket_stream(query: str, api_base: str, api_key: str) -> Tuple[str, str, str, str, str, str]:
    query = (query or "").strip()
    if not query:
        return (
            "",
            "",
            "",
            "",
            "",
            json.dumps({"mode": "websocket", "error": "Enter a question."}),
        )
    base = (api_base or _api_base()).rstrip("/")
    return asyncio.run(_ws_collect(base, query, api_key))


def build_demo():
    with gr.Blocks(
        title="Transparent Research Assistant",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"),
        css="""
        .sandbox-console textarea { font-family: ui-monospace, monospace; font-size: 0.85rem; }
        .consensus-panel { border-radius: 8px; padding: 0.5rem 1rem; }
        """,
    ) as demo:
        gr.Markdown(
            "## Transparent Research UI\n"
            "Glass-box chat: **thinking** (ChainOfThoughtBuilder), **consensus** status, "
            "**sources** with trust proxies, and **sandbox** console hints.\n\n"
            "Start the API: `uvicorn api.main:app --reload`"
        )
        with gr.Row():
            api_base = gr.Textbox(label="API base URL", value=_api_base())
            api_key = gr.Textbox(label="X-API-Key (if API_KEY is set on server)", type="password")
            session_id = gr.Textbox(label="Session ID (optional ResearchLog)", placeholder="")

        mode = gr.Radio(
            choices=[
                ("Transparent (REST) — full pipeline metadata", "rest"),
                ("Stream (WebSocket) — live tokens + CoT/sources first", "ws"),
            ],
            value="rest",
            label="Mode",
        )

        query = gr.Textbox(label="Your question", lines=3, placeholder="Ask a grounded research question…")
        submit = gr.Button("Run", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Thinking trace (ChainOfThoughtBuilder / scratchpad)", open=True):
                    thinking_out = gr.Markdown()
                gr.Markdown("### Answer")
                answer_out = gr.Markdown()
                with gr.Accordion("Consensus & verification JSON", open=False):
                    consensus_detail = gr.Code(language="json")
            with gr.Column(scale=1):
                gr.Markdown("### Source sidecar (FAISS / retrieval)")
                sources_out = gr.Markdown()
                gr.Markdown("### Consensus indicator")
                consensus_badge = gr.Markdown(elem_classes=["consensus-panel"])

        gr.Markdown("### Sandbox console")
        sandbox_out = gr.Textbox(
            lines=10,
            max_lines=20,
            label="STDOUT / execution hints",
            elem_classes=["sandbox-console"],
        )

        status = gr.Markdown()

        def _submit(q, base, key, sess, m):
            if m == "ws":
                t, s, b, a, sb, d = run_websocket_stream(q, base, key)
                return t, s, b, a, sb, d, ""
            t, s, b, a, sb, d = run_transparent_rest(q, base, key, sess)
            return t, s, b, a, sb, d, ""

        submit.click(
            fn=_submit,
            inputs=[query, api_base, api_key, session_id, mode],
            outputs=[
                thinking_out,
                sources_out,
                consensus_badge,
                answer_out,
                sandbox_out,
                consensus_detail,
                status,
            ],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name=os.getenv("RESEARCH_UI_HOST", "127.0.0.1"), server_port=int(os.getenv("RESEARCH_UI_PORT", "7861")))


if __name__ == "__main__":
    main()
