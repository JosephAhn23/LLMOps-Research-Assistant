# Live Demo Checklist

This checklist is for turning the local project into a hiring-manager-ready live demo.

| Step | Command / Action | Evidence to capture |
|:---|:---|:---|
| 1 | `docker compose up -d` | Services healthy screenshot |
| 2 | `uvicorn api.main:app --host 0.0.0.0 --port 8000` | API startup log |
| 3 | Run one query through `/query` endpoint | JSON response with citations |
| 4 | `python benchmarks/run_ragas.py` | `mlops/ragas_baseline.json` output |
| 5 | `python benchmarks/run_benchmarks.py` | `benchmarks/results.json` output |
| 6 | Record 20-30s GIF (query -> logs -> answer) and save `assets/quick_demo.gif` | Embedded GIF in README |
| 7 | Deploy API to target host (Render/Fly.io/AWS/GCP) and update `Live Demo` badge URL in `README.md` | Public URL + working endpoint |

## Demo script (20-30 seconds)

```text
1) Ask: "How does reranking improve RAG quality?"
2) Show retrieval + reranking timings in logs.
3) Show grounded answer with citations.
4) Close on headline metric banner in README.
```

## Definition of done

| Requirement | Done when |
|:---|:---|
| Public URL | `README.md` Live Demo badge points to deployed app |
| Reproducibility | `RESULTS.md` commands run successfully from clean clone |
| Proof | GIF + benchmark artifacts are committed and visible |

