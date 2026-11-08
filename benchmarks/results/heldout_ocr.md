# Results — E (OCR)

- Generated: 2026-11-08T20:40:56+00:00
- Model: `BAAI/bge-small-en-v1.5`
- Mode: `E`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 4 across 3 videos
- Query file SHA-256: `64b80d920204d73d…`

| Metric | Value |
|---|---|
| Recall@1 | 0.250 |
| Recall@3 | 0.250 |
| Recall@5 | 0.500 |
| Recall@10 | 0.500 |
| mIoU (segments) | 0.062 |
| Duplicate rate (raw top-10) | 0.050 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 82 ms |
| Latency p95 | 303 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| ocr | 4 | 0.250 | 0.250 | 0.500 | 0.500 | 0.062 | 0.050 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| lecture1 | 1 | 0.000 | 0.000 | 0.000 |
| lecture2 | 1 | 0.000 | 0.000 | 0.000 |
| screen-recording | 2 | 0.500 | 1.000 | 0.125 |
