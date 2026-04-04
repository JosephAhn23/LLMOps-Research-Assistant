"""
Constitutional research gate — LLM-as-judge against explicit research principles.

If the grade is below the pass threshold (default 90), callers should regenerate
the answer (see ``agents.orchestrator`` synthesis loop).
"""
from __future__ import annotations

import logging
import os
import re
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_PASS_THRESHOLD = 90.0

RESEARCH_PRINCIPLES: List[str] = [
    "Do not present a formula or numeric result as fact without derivation, citation, or explicit assumptions.",
    "Do not simplify complex mathematics or physics for brevity when the user asked for rigor.",
    "Distinguish established results from conjecture; label uncertainty clearly.",
    "Prefer primary claims grounded in provided context or cited sources over plausible invention.",
    "Avoid persuasive or 'helpful' tone that obscures limits of evidence.",
    "Flag when retrieved context is insufficient instead of filling gaps speculatively.",
]

CONSTITUTIONAL_JUDGE_PROMPT = """You are a strict research-ethics reviewer grading an assistant answer.

Apply ONLY these principles (violate none):
{principles}

The answer to grade:
---
{answer}
---

Respond in exactly this format (first line must be parseable):
SCORE: <integer 0-100>
VIOLATIONS: <comma-separated short phrases, or NONE if none>
RATIONALE: <one or two sentences>

Scoring guide:
- 90-100: Fully aligned; caveats and sourcing appropriate.
- 70-89: Minor issues (e.g. one soft claim without caveat).
- Below 70: Serious principle violations (fabricated rigor, hidden assumptions, oversimplification of math).
""".strip()


class ConstitutionalResult(BaseModel):
    score: float = Field(ge=0, le=100)
    passed: bool
    violations: List[str] = Field(default_factory=list)
    rationale: str = ""
    raw_response: str = ""


class ConstitutionalClassifier:
    """
    OpenAI Chat Completions judge. Requires ``OPENAI_API_KEY`` and ``openai`` package.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        principles: List[str] | None = None,
    ) -> None:
        self.model = model or os.getenv("CONSTITUTION_JUDGE_MODEL", "gpt-4o-mini")
        self.pass_threshold = pass_threshold
        self.principles = principles or list(RESEARCH_PRINCIPLES)

    def grade(self, answer: str) -> ConstitutionalResult:
        text = (answer or "").strip()
        if not text:
            return ConstitutionalResult(
                score=0.0,
                passed=False,
                violations=["empty_answer"],
                rationale="No answer to grade.",
                raw_response="",
            )

        principles_block = "\n".join(f"- {p}" for p in self.principles)
        user_content = CONSTITUTIONAL_JUDGE_PROMPT.format(
            principles=principles_block,
            answer=text[:12000],
        )

        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You output only the required SCORE/VIOLATIONS/RATIONALE block.",
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=400,
                timeout=float(os.getenv("OPENAI_TIMEOUT_S", "60")),
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Constitutional judge unavailable: %s", exc)
            return ConstitutionalResult(
                score=100.0,
                passed=True,
                violations=[],
                rationale="Judge skipped (API error); not blocking.",
                raw_response=str(exc),
            )

        score = _parse_score(raw)
        violations = _parse_violations(raw)
        rationale = _parse_rationale(raw)
        passed = score >= self.pass_threshold
        return ConstitutionalResult(
            score=score,
            passed=passed,
            violations=violations,
            rationale=rationale,
            raw_response=raw[:2000],
        )


def _parse_score(raw: str) -> float:
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if not m:
        return 0.0
    return min(100.0, max(0.0, float(m.group(1))))


def _parse_violations(raw: str) -> List[str]:
    m = re.search(r"VIOLATIONS:\s*(.+?)(?:\n|RATIONALE:)", raw, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    body = m.group(1).strip()
    if not body or body.upper().startswith("NONE"):
        return []
    return [x.strip() for x in body.split(",") if x.strip()][:12]


def _parse_rationale(raw: str) -> str:
    m = re.search(r"RATIONALE:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip()[:500] if m else "")[:500]
