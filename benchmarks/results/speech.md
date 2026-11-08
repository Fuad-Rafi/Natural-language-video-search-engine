# Results — D (speech)

- Generated: 2026-11-08T18:02:41+00:00
- Model: `BAAI/bge-small-en-v1.5`
- Mode: `D`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 9 across 2 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.778 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |
| mIoU (segments) | 0.310 |
| Duplicate rate (raw top-10) | 0.000 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 17 ms |
| Latency p95 | 1238 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| speech | 9 | 0.778 | 1.000 | 1.000 | 1.000 | 0.310 | 0.000 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| lecture1 | 5 | 0.600 | 1.000 | 0.268 |
| lecture2 | 4 | 1.000 | 1.000 | 0.363 |
