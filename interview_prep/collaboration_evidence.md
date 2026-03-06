# Collaboration Evidence

## How I Demonstrate Team Readiness From This Repo

I treat this project as if it were a shared codebase, not a solo sandbox. The point is to show reviewer empathy, change isolation, and explicit trade-off communication.

## 1) Self-Review Checklist (Used Before Any PR)

| Review dimension | What I check |
|:---|:---|
| Behavioral diff | "What user behavior changed?" not just "what files changed." |
| Risk surface | Failure modes, rollback path, and backward compatibility. |
| Observability | Logs, metrics, and traces needed to debug in production. |
| Validation | Unit/integration tests plus any benchmark impact. |
| Decision log | Why this approach was chosen over alternatives. |

## 2) Example PR Review Comments I Would Leave

```text
Blocking:
- `agents/multi_agent/supervisor.py`: retry path catches exceptions but does not
  increment a separate "degraded_response" metric. We need that to know if users
  are getting fallback answers at elevated rates.

Non-blocking:
- `README.md`: cost table should include sample size and measurement method in the
  footnote. Raw percentages without context invite skepticism in interviews.

Question:
- In `ingestion/pipeline.py`, lock scope includes embedding calls. Did we profile
  contention under concurrent ingest, or should lock only protect index write?
```

## 3) Feedback-Handling Framework

| Feedback type | My response pattern |
|:---|:---|
| Correctness bug | Acknowledge, reproduce, add regression test, patch quickly. |
| Scope concern | Split PR into smaller deltas and move extras to follow-up issue. |
| Style disagreement | Align to project standard unless it harms readability. |
| Product disagreement | Restate objective, compare options, document decision rationale. |

## 4) Public Signals I Will Add

1. Open issues with labels (`bug`, `good first issue`, `needs-data`).
2. Draft PRs with explicit "requesting design feedback" sections.
3. At least one external collaborator review on non-trivial changes.
4. Weekly changelog notes summarizing impact and risks, not just file lists.

