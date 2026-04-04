# Case Study 02: Experiment Analysis for Reranker-v2 Rollout

## Objective

Decide whether `reranker-v2` should be shipped to 100% traffic using statistically valid experimentation and causal inference.

| Field | Value |
|:---|:---|
| Decision question | Does `reranker-v2` improve answer quality without violating latency guardrails? |
| Experiment type | 50/50 online A/B with sequential looks |
| Primary metric | `ragas_faithfulness` |
| Guardrail | `p95_latency_ms` must not degrade by > `+8%` |
| Analysis stack | SRM -> CUPED -> O'Brien-Fleming -> Double ML -> markdown report |

## Mock Data Snapshot

| Metric | Control | Treatment |
|:---|---:|---:|
| Users assigned | 51,176 | 50,824 |
| Sessions analyzed | 18,420 | 18,311 |
| RAGAS faithfulness mean | 0.836 | 0.851 |
| Answer relevancy mean | 0.818 | 0.829 |
| p95 latency (ms) | 3,410 | 3,498 |
| Avg tokens/query | 1,612 | 1,645 |

Data generation assumptions: 7-day run, English-only support/research traffic, one answer per session, outliers above 30s removed (`n_removed=132`).

## 1) SRM Detection

Observed split is checked with chi-squared against expected 50/50 assignment.

| Test | Value |
|:---|:---|
| Expected split | 50% / 50% |
| Observed split | 50.17% / 49.83% |
| Chi-squared statistic | `0.124` |
| p-value | `0.724` |
| Decision | No SRM detected (assignment mechanism healthy) |

## 2) CUPED Variance Reduction

Pre-experiment covariate: user-level baseline answer satisfaction score over prior 14 days.

| Metric | Before CUPED | After CUPED | Improvement |
|:---|---:|---:|---:|
| Variance (`ragas_faithfulness`) | 0.0142 | 0.0090 | `-36.6%` |
| Required sample for MDE=1.0% | 39,800 | 25,300 | `-36.4%` |
| Theta coefficient | - | `0.418` | - |

## 3) O'Brien-Fleming Sequential Test

Planned 4 looks at 25%, 50%, 75%, and 100% information fractions.

| Look | Information Fraction | Z-stat | Boundary | Decision |
|:---:|---:|---:|---:|:---|
| 1 | 0.25 | 1.18 | 2.80 | Continue |
| 2 | 0.50 | 1.94 | 2.40 | Continue |
| 3 | 0.75 | 2.33 | 2.13 | Crossed boundary (early win) |
| 4 | 1.00 | 2.41 | 1.99 | Confirmed win |

Result: reject `H0` at look 3 without inflating Type-I error.

## 4) Double ML Causal Estimate

Used k-fold cross-fitting (`k=5`) with gradient boosting nuisance models on user/device/region/time covariates.

| Estimate | Value |
|:---|:---|
| ATE on faithfulness | `+0.013` |
| 95% CI | `[+0.006, +0.020]` |
| p-value | `0.001` |
| Interpretation | Positive causal impact after adjusting for confounders |

## 5) Automated Markdown Report (Output)

```markdown
# Experiment Report: reranker_v2_rollout

Decision: SHIP
Primary effect: ragas_faithfulness +1.8% (CUPED-adjusted), p=0.0032
Sequential status: crossed O'Brien-Fleming efficacy boundary at look 3
Guardrails: PASS (latency +2.6%, below +8% threshold)

Business impact:
- Estimated 124 fewer unsupported answers per 10,000 responses.
- Projected +4.1% reduction in user follow-up queries.
```

## Final Decision

| Dimension | Result |
|:---|:---|
| Statistical validity | Pass (no SRM, sequential alpha control) |
| Effect magnitude | Positive and practically meaningful |
| Causal robustness | Positive ATE with non-zero lower CI bound |
| Guardrails | Pass |
| Recommendation | Ship `reranker-v2` to 100% with post-launch monitoring |

