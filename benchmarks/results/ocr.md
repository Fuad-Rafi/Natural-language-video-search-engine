# Results — E (OCR)

- Generated: 2026-11-08T20:39:39+00:00
- Model: `BAAI/bge-small-en-v1.5`
- Mode: `E`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 12 across 3 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.583 |
| Recall@3 | 0.750 |
| Recall@5 | 0.750 |
| Recall@10 | 0.833 |
| mIoU (segments) | 0.270 |
| Duplicate rate (raw top-10) | 0.117 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 24 ms |
| Latency p95 | 36 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| ocr | 12 | 0.583 | 0.750 | 0.750 | 0.833 | 0.270 | 0.117 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| lecture1 | 3 | 0.667 | 0.667 | 0.500 |
| lecture2 | 3 | 0.000 | 0.333 | 0.000 |
| screen-recording | 6 | 0.833 | 1.000 | 0.290 |
