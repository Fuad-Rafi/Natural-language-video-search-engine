# Results — A (frames)

- Generated: 2026-11-08T15:14:30+00:00
- Model: `google/siglip-base-patch16-224`
- Mode: `A`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 20 across 8 videos
- Query file SHA-256: `64b80d920204d73d…`

| Metric | Value |
|---|---|
| Recall@1 | 0.550 |
| Recall@5 | 0.600 |
| Recall@10 | 0.700 |
| mIoU (segments) | 0.137 |
| Duplicate rate (raw top-10) | 0.450 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 12 ms |
| Latency p95 | 25 ms |

## Per query type

| Type | Queries | R@1 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|
| action | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| caption | 1 | 1.000 | 1.000 | 1.000 | 0.188 | 0.600 | 0.000 |
| object | 5 | 0.600 | 0.600 | 0.800 | 0.122 | 0.340 | 0.000 |
| ocr | 4 | 1.000 | 1.000 | 1.000 | 0.356 | 0.575 | 0.000 |
| speech | 4 | 0.500 | 0.500 | 0.750 | 0.074 | 0.550 | 0.000 |
| visual | 5 | 0.200 | 0.400 | 0.400 | 0.045 | 0.340 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 3 | 0.333 | 0.333 | 0.067 |
| indoor1 | 2 | 0.500 | 0.500 | 0.094 |
| indoor2 | 2 | 0.500 | 1.000 | 0.050 |
| lecture1 | 3 | 0.667 | 0.667 | 0.238 |
| lecture2 | 3 | 0.667 | 0.667 | 0.116 |
| screen-recording | 2 | 1.000 | 1.000 | 0.330 |
| streetcctv1 | 3 | 0.667 | 0.667 | 0.178 |
| streetcctv2 | 2 | 0.000 | 0.000 | 0.000 |
