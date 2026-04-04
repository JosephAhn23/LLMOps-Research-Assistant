# Interview Prep Artifacts

This folder exists to show the human side of the project: decisions, communication patterns, and proof that I can operate in real engineering environments instead of only shipping solo code.

| Artifact | Why it matters in interviews |
|:---|:---|
| `oss_contribution_plan.md` | Shows I can read/modify external codebases and work with maintainers. |
| `collaboration_evidence.md` | Shows how I review, receive feedback, and write decision-oriented PR communication. |
| `linkedin_profile_copy.md` | Keeps recruiter-facing story aligned with the repo evidence and metrics. |

Use these documents as talking points, not scripts.

## Technical Phone Screen Q&A (Drill These Out Loud)

### Q1) Why did you choose FAISS over Pinecone?
I chose FAISS because the bottleneck in this workload was generation latency, not retrieval latency. In local benchmarks (`n=50` queries), retrieval stayed under 5 ms and reranking was roughly 40-47 ms, while end-to-end time was dominated by LLM round-trip time in seconds. A managed vector DB would have added network hops and another service bill without fixing the core latency constraint. The trade-off was operational burden: I had to own index persistence and metadata consistency in `ingestion/pipeline.py`. If this were high-velocity, multi-tenant production with strict uptime SLOs, I would likely revisit that choice.

### Q2) Walk me through what happens when the circuit breaker opens.
In `agents/multi_agent/supervisor.py`, each agent has a `CircuitBreaker` from `agents/multi_agent/failure_handling.py`. During `_run_agent_safe`, if `allow_call()` returns false, the supervisor does not keep retrying blindly; it returns a degraded `AgentResult` with `CIRCUIT_OPEN` status immediately. That protects end-to-end latency and avoids a cascading failure when one agent is unhealthy. The pipeline still proceeds to consensus/fallback logic so the user gets a response path instead of total failure. Once healthy signals return, the breaker can recover through its state transitions rather than staying permanently open.

### Q3) What was the hardest bug you hit in the multi-agent system?
The hardest class of bugs was not model quality, it was reliability under partial failure. A failing agent could repeatedly timeout and drag total latency far beyond acceptable ranges if retries were naive. I fixed this by combining retries with circuit-breaker guardrails so failure behavior is bounded. I also made sure we preserve observability with per-agent tracing so I can distinguish "bad output" from "agent unavailable" paths. The lesson was that in multi-agent systems, failure-mode design matters as much as generation quality.

### Q4) How do you detect retrieval quality degradation in production?
I track answer quality with RAGAS metrics in `mlops/ragas_tracker.py` and compare runs against a saved baseline (`mlops/ragas_baseline.json`). The CI gate in `cicd/ragas_gate.py` blocks merges when metric deltas exceed configured thresholds, so regressions are caught before deployment. In practice, I pay closest attention to faithfulness drops first (hallucination risk), then relevancy drops (user usefulness risk). I also tie quality checks to the same retrieval/reranking pipeline so regressions are measured in realistic system context, not isolated toy prompts.

### Q5) How would you explain this repo without sounding generic?
I would anchor on one request path: `api/main.py` receives a query, `agents/orchestrator.py` drives retrieval/reranking/synthesis, and `agents/multi_agent/supervisor.py` handles resilience and traceability. Then I call out one concrete trade-off (FAISS ownership vs managed convenience), one concrete reliability decision (circuit breakers over naive retries), and one concrete quality gate (RAGAS baseline regression in CI). I avoid broad claims unless I can cite the exact file and benchmark method behind the number. If I cannot explain a directory’s failure mode or trade-off, I treat that as a gap to close before interviews.

