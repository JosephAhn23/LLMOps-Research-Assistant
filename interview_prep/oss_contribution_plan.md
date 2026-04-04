# OSS Contribution Plan (30 Days)

## Goal

Land one merged pull request in a production-grade LLM toolchain project to prove maintainership-level collaboration skills.

## Target Repositories

| Priority | Project | Why this is a fit for this repo |
|:---:|:---|:---|
| 1 | LangChain / LangGraph | I already use graph orchestration and tracing patterns in `agents/orchestrator.py` and `agents/multi_agent/supervisor.py`. |
| 2 | RAGAS | My CI quality gate and baseline workflow already depend on RAGAS metrics in `mlops/ragas_tracker.py`. |
| 3 | MLflow | I log eval metrics and regressions to MLflow; good fit for usability/docs fixes. |

## Contribution Strategy

| Week | Action | Output |
|:---:|:---|:---|
| 1 | Read contribution docs, set up local test/lint, shortlist 5 "good first issue" items. | Public issue shortlist in GitHub project board. |
| 2 | Submit 2 small PRs (docs/tests/typing bugfixes) with passing CI. | 2 opened PRs with maintainer feedback loops. |
| 3 | Submit 1 medium PR touching implementation and tests. | 1 merged PR target or review-ready iteration. |
| 4 | Publish a short write-up of what changed and what reviewer feedback taught me. | `docs/oss_contribution_retrospective.md` |

## PR Selection Criteria

1. Reproducible issue with failing test or docs gap tied to observable behavior.
2. Scope small enough for review in one sitting (roughly under 300 lines changed).
3. Change includes tests or benchmark evidence, not only code edits.
4. I can explain trade-offs and compatibility impact in the PR description.

## PR Template I Will Use

```text
Problem:
- What user-facing behavior is wrong or unclear?

Root cause:
- Why it happens in current code path.

Fix:
- Exact code-level behavior change.

Validation:
- Tests run, benchmark delta, and backward-compatibility notes.

Risk:
- What could regress and why it is low/acceptable.
```

## Success Criteria

| Metric | Target |
|:---|:---|
| Opened OSS PRs | >= 3 in 30 days |
| Merged OSS PRs | >= 1 in 30 days |
| Review cycles handled | >= 2 rounds on at least one PR |
| Public retrospective | 1 published note with lessons learned |

