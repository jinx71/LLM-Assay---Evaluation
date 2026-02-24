# LLM Assay — Evaluation Report

- **Run:** 2026-06-26T06:19:17.899332+00:00
- **Datasets:** gmp_severity, sop_extraction
- **Evaluations:** 50
- **Duration:** 0.6s

## Leaderboard

| # | Model | Score | Pass | Mean ms | p95 ms | Cost $ | Err |
|---|-------|------:|-----:|--------:|-------:|-------:|----:|
| 1 | `mock:smart` | 26.7% | 24.0% | 36 | 49 | 0.0000 | 0% |
| 2 | `mock:weak` | 8.0% | 8.0% | 83 | 110 | 0.0000 | 0% |

## Dataset: gmp_severity

| Model | Score | Pass | Mean ms | Cost $ |
|-------|------:|-----:|--------:|-------:|
| `mock:smart` | 40.0% | 40.0% | 38 | 0.0000 |
| `mock:weak` | 13.3% | 13.3% | 80 | 0.0000 |

## Dataset: sop_extraction

| Model | Score | Pass | Mean ms | Cost $ |
|-------|------:|-----:|--------:|-------:|
| `mock:smart` | 6.7% | 0.0% | 34 | 0.0000 |
| `mock:weak` | 0.0% | 0.0% | 88 | 0.0000 |
