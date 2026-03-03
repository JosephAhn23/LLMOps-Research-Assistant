# Operability Evidence Pack

Use this checklist to convert the project from "good architecture" into "production credibility" for interviews, portfolio reviews, and hiring loops.

## 1) SLO Definitions

Fill this table with your measured numbers from benchmark runs.

| SLI | Target SLO | Measurement Method | Current |
|---|---|---|---|
| Query success rate | >= 99.5% | API benchmark success ratio | TODO |
| p95 latency (`/query`) | <= 1500 ms | `analysis/repro_benchmark.py --mode api` | TODO |
| Batch completion rate | >= 99% | Celery job completion ratio | TODO |
| Error budget burn | <= 1%/month | CloudWatch error counts | TODO |

## 2) Reproducible Benchmark Artifacts

Run:

```bash
python -m analysis.repro_benchmark --mode offline_pipeline --runs 200 --concurrency 20
python -m analysis.repro_benchmark --mode api --runs 200 --concurrency 20 --api-url http://127.0.0.1:8000/query
```

Artifacts are written to `analysis/benchmarks/<timestamp>/`:
- `metrics.json`
- `summary.md`

Capture one "baseline" folder and one "after optimization" folder for an A/B comparison.

## 3) Failure-Mode Drills

Document at least one run for each scenario:

1. **Shard outage**
   - Stop one shard service.
   - Confirm aggregator still returns partial results.
   - Record impact to success rate and p95 latency.
2. **Latency spike**
   - Inject delay on one shard.
   - Confirm global top-k still returns results.
   - Measure p95 regression.
3. **Dependency outage**
   - Disable model/reranker backend.
   - Verify graceful fallback behavior and alerting.

Attach logs/plots for each drill in `analysis/benchmarks/` or `docs/evidence/`.

## 4) Rollback Drill Template

Track one real rollback rehearsal and keep this updated.

- Deployment identifier: `TODO`
- Trigger condition: `TODO`
- Rollback command(s): `TODO`
- Recovery time objective (RTO): `TODO`
- Actual recovery time: `TODO`
- Post-rollback verification checks:
  - [ ] `/health` endpoint green
  - [ ] Error rate back to baseline
  - [ ] Query quality checks pass
  - [ ] Batch queue drains normally

## 5) Monitoring Evidence Checklist

Collect screenshots (or exported dashboards) that prove runtime health:

- [ ] API p50/p95 latency
- [ ] Request throughput
- [ ] Error rate / exceptions
- [ ] Celery queue depth and worker throughput
- [ ] CPU/memory for API + workers + shard services
- [ ] MLflow experiment comparison for key model/retrieval changes

Suggested storage path: `docs/evidence/<date>/`.
