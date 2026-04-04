"""
Policy enforcement — constitution-style rules + drop claims unsupported by RAG chunks.

Conservative default: remove sentences that neither cite ``[source_N]`` with a valid
index nor show lexical overlap with retrieved chunk text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.multi_agent.base_agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    BaseAgent,
    ToolRegistry,
)

_CITATION = re.compile(r"\[source_(\d+)\]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


DEFAULT_CONSTITUTION = [
    "Never state organization-specific policy unless it appears in the provided RAG context.",
    "Do not invent citations, URLs, or document titles.",
    "If evidence is missing, say the policy or fact is not in context instead of guessing.",
]


@dataclass
class PolicyEnforcementResult:
    enforced_answer: str
    dropped_sentences: List[str] = field(default_factory=list)
    kept_count: int = 0


def enforce_grounded_answer(
    answer: str,
    chunks: List[Dict[str, Any]],
    *,
    constitution: Optional[List[str]] = None,
    overlap_threshold: float = 0.08,
) -> PolicyEnforcementResult:
    """
    Filter sentences: keep if valid [source_k] in sentence or overlap with some chunk.
    Short sentences (< 10 words) that are transitional are kept.
    """
    _ = constitution  # reserved for LLM judge expansion; rules applied structurally here
    if not (answer or "").strip():
        return PolicyEnforcementResult("", [], 0)

    n_chunks = len(chunks)

    def chunk_tokens(i: int) -> set[str]:
        if i < 0 or i >= n_chunks:
            return set()
        t = chunks[i].get("text") or ""
        return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", t)}

    all_chunk_tokens: List[set[str]] = [chunk_tokens(i) for i in range(n_chunks)]

    def sentence_supported(sent: str) -> bool:
        s = sent.strip()
        if len(s.split()) < 10 and not re.search(r"\b(shall|must|will|policy|guarantee)\b", s, re.I):
            return True
        m = _CITATION.search(s)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_chunks:
                st = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", s)}
                if not st:
                    return True
                inter = len(st & all_chunk_tokens[idx])
                return inter >= 1 or len(st) < 4
            return False
        st = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", s)}
        if len(st) < 3:
            return True
        for cset in all_chunk_tokens:
            if not cset:
                continue
            j = len(st & cset) / max(len(st | cset), 1)
            if j >= overlap_threshold:
                return True
        return False

    kept: List[str] = []
    dropped: List[str] = []
    for part in _SENT_SPLIT.split(answer.strip()):
        if not part.strip():
            continue
        if sentence_supported(part):
            kept.append(part)
        else:
            dropped.append(part[:500])

    enforced = " ".join(kept).strip()
    if dropped and not enforced:
        enforced = (
            "[Policy enforcement] All substantive claims were removed because they lacked "
            "support in the retrieved context. Ask a narrower question or ingest relevant policy."
        )

    return PolicyEnforcementResult(
        enforced_answer=enforced,
        dropped_sentences=dropped,
        kept_count=len(kept),
    )


class PolicyEnforcementAgent(BaseAgent):
    """
    Multi-agent policy gate: constitution + structural grounding filter.
    """

    def __init__(
        self,
        name: str = "policy_enforcement",
        tools: Optional[ToolRegistry] = None,
        constitution: Optional[List[str]] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(name, tools, timeout_seconds)
        self.constitution = constitution or list(DEFAULT_CONSTITUTION)

    def process(self, task: AgentTask) -> AgentResult:
        draft = (
            task.context.get("draft_answer")
            or task.context.get("research_result")
            or ""
        )
        chunks = task.context.get("retrieved_context") or task.context.get("chunks") or []

        if not draft.strip():
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error="No draft answer to enforce.",
                confidence=0.0,
            )

        res = enforce_grounded_answer(draft, chunks, constitution=self.constitution)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=AgentStatus.SUCCEEDED,
            output=res.enforced_answer,
            confidence=1.0 if not res.dropped_sentences else 0.75,
            reasoning=f"Dropped {len(res.dropped_sentences)} unsupported sentence(s).",
            metadata={
                "enforced_answer": res.enforced_answer,
                "dropped_sentences": res.dropped_sentences,
                "constitution": self.constitution,
            },
        )
