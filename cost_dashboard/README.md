# Cost Dashboard

Tracks per-query cost and summarizes operating spend across model options.

## Supported pricing modes

| Mode | How cost is computed |
|:---|:---|
| `gpt-4o` | Retrieval + generation token rates from built-in defaults |
| `gpt-4o-mini` | Retrieval + generation token rates from built-in defaults |
| `self_hosted` | Uses configurable `$ / 1k tokens` for retrieval + generation |

Reranking compute is tracked separately as direct USD per query.

## Example output

| Metric | Value |
|:---|:---|
| Queries tracked | `10,000` |
| Avg cost / query | `$0.0027` |
| p50 cost / query | `$0.0024` |
| p99 cost / query | `$0.0051` |
| Projected monthly cost @ 2 QPS | `$13,996.80` |

## Example usage

```python
from cost_dashboard.tracker import CostTracker

tracker = CostTracker()
tracker.log_query(
    model="gpt-4o-mini",
    retrieval_tokens=380,
    generation_input_tokens=920,
    generation_output_tokens=210,
    reranking_compute_usd=0.0008,
)
summary = tracker.summarize(n_qps=2.0)
print(summary)
```

