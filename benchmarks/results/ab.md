# Results — A+B (RRF)

- Generated: 2026-11-08T17:59:11+00:00
- Model: `google/siglip-base-patch16-224`
- Mode: `AB`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 50 across 8 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.620 |
| Recall@3 | 0.780 |
| Recall@5 | 0.880 |
| Recall@10 | 0.900 |
| mIoU (segments) | 0.242 |
| Duplicate rate (raw top-10) | 0.384 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 13 ms |
| Latency p95 | 23 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| action | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.475 | 0.460 | 0.000 |
| caption | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.227 | 0.500 | 0.000 |
| counting | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.571 | 0.600 | 0.000 |
| object | 9 | 0.556 | 0.778 | 1.000 | 1.000 | 0.147 | 0.178 | 0.000 |
| ocr | 12 | 0.750 | 0.833 | 0.917 | 0.917 | 0.354 | 0.583 | 0.000 |
| spatial | 2 | 0.500 | 1.000 | 1.000 | 1.000 | 0.025 | 0.000 | 0.000 |
| speech | 9 | 0.333 | 0.556 | 0.556 | 0.667 | 0.096 | 0.311 | 0.000 |
| visual | 11 | 0.545 | 0.727 | 0.909 | 0.909 | 0.221 | 0.400 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 7 | 0.429 | 1.000 | 0.234 |
| indoor1 | 4 | 0.750 | 1.000 | 0.493 |
| indoor2 | 4 | 1.000 | 1.000 | 0.444 |
| lecture1 | 8 | 0.500 | 0.750 | 0.210 |
| lecture2 | 7 | 0.429 | 0.571 | 0.128 |
| screen-recording | 8 | 0.750 | 1.000 | 0.366 |
| streetcctv1 | 6 | 0.667 | 1.000 | 0.079 |
| streetcctv2 | 6 | 0.667 | 0.833 | 0.119 |
