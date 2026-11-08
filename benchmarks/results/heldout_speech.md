# Results — D (speech)

- Generated: 2026-11-08T18:03:45+00:00
- Model: `BAAI/bge-small-en-v1.5`
- Mode: `D`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 4 across 2 videos
- Query file SHA-256: `64b80d920204d73d…`

| Metric | Value |
|---|---|
| Recall@1 | 0.750 |
| Recall@3 | 0.750 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |
| mIoU (segments) | 0.311 |
| Duplicate rate (raw top-10) | 0.000 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 18 ms |
| Latency p95 | 280 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| speech | 4 | 0.750 | 0.750 | 1.000 | 1.000 | 0.311 | 0.000 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| lecture1 | 2 | 0.500 | 1.000 | 0.093 |
| lecture2 | 2 | 1.000 | 1.000 | 0.529 |
