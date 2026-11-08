# Results — B (objects)

- Generated: 2026-11-08T15:15:17+00:00
- Model: `google/siglip-base-patch16-224`
- Mode: `B`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 20 across 8 videos
- Query file SHA-256: `64b80d920204d73d…`

| Metric | Value |
|---|---|
| Recall@1 | 0.150 |
| Recall@5 | 0.450 |
| Recall@10 | 0.550 |
| mIoU (segments) | 0.042 |
| Duplicate rate (raw top-10) | 0.148 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 22 ms |
| Latency p95 | 201 ms |

## Per query type

| Type | Queries | R@1 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|
| action | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.100 | 0.000 |
| caption | 1 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| object | 5 | 0.400 | 0.600 | 0.600 | 0.127 | 0.150 | 0.000 |
| ocr | 4 | 0.000 | 0.250 | 0.250 | 0.000 | 0.075 | 0.000 |
| speech | 4 | 0.000 | 0.250 | 0.250 | 0.000 | 0.225 | 0.000 |
| visual | 5 | 0.200 | 0.600 | 0.800 | 0.040 | 0.180 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 3 | 0.000 | 0.333 | 0.000 |
| indoor1 | 2 | 0.000 | 0.500 | 0.000 |
| indoor2 | 2 | 0.500 | 0.500 | 0.100 |
| lecture1 | 3 | 0.000 | 0.000 | 0.000 |
| lecture2 | 3 | 0.000 | 0.667 | 0.000 |
| screen-recording | 2 | 0.000 | 0.000 | 0.000 |
| streetcctv1 | 3 | 0.333 | 0.667 | 0.012 |
| streetcctv2 | 2 | 0.500 | 1.000 | 0.300 |
