# Results — B (objects)

- Generated: 2026-11-08T17:58:27+00:00
- Model: `google/siglip-base-patch16-224`
- Mode: `B`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 50 across 8 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.220 |
| Recall@3 | 0.340 |
| Recall@5 | 0.440 |
| Recall@10 | 0.520 |
| mIoU (segments) | 0.042 |
| Duplicate rate (raw top-10) | 0.112 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 26 ms |
| Latency p95 | 227 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| action | 5 | 0.600 | 0.600 | 0.800 | 1.000 | 0.151 | 0.080 | 0.000 |
| caption | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.100 | 0.000 |
| counting | 1 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.300 | 0.000 |
| object | 9 | 0.444 | 0.667 | 0.778 | 0.889 | 0.089 | 0.133 | 0.000 |
| ocr | 12 | 0.083 | 0.083 | 0.167 | 0.167 | 0.003 | 0.092 | 0.000 |
| spatial | 2 | 0.500 | 1.000 | 1.000 | 1.000 | 0.025 | 0.200 | 0.000 |
| speech | 9 | 0.000 | 0.111 | 0.111 | 0.111 | 0.000 | 0.189 | 0.000 |
| visual | 11 | 0.182 | 0.273 | 0.364 | 0.545 | 0.043 | 0.036 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 7 | 0.000 | 0.286 | 0.000 |
| indoor1 | 4 | 0.750 | 1.000 | 0.179 |
| indoor2 | 4 | 0.500 | 1.000 | 0.040 |
| lecture1 | 8 | 0.000 | 0.000 | 0.000 |
| lecture2 | 7 | 0.143 | 0.429 | 0.005 |
| screen-recording | 8 | 0.000 | 0.000 | 0.000 |
| streetcctv1 | 6 | 0.667 | 0.833 | 0.185 |
| streetcctv2 | 6 | 0.167 | 0.667 | 0.017 |
