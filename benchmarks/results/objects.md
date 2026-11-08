# Baseline results — B (objects)

- Generated: 2026-11-08T14:54:01+00:00
- Model: `google/siglip-base-patch16-224`
- Layer: `objects`
- Queries: 50 across 8 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.200 |
| Recall@5 | 0.340 |
| Recall@10 | 0.480 |
| mIoU (1 s window) | 0.014 |
| Latency p50 | 9 ms |
| Latency p95 | 11 ms |

## Per query type

| Type | Queries | R@1 | R@5 | R@10 | mIoU |
|---|---|---|---|---|---|
| action | 5 | 0.600 | 0.600 | 1.000 | 0.047 |
| caption | 1 | 0.000 | 1.000 | 1.000 | 0.000 |
| counting | 1 | 0.000 | 0.000 | 1.000 | 0.000 |
| object | 9 | 0.444 | 0.778 | 0.778 | 0.040 |
| ocr | 12 | 0.083 | 0.083 | 0.167 | 0.003 |
| spatial | 2 | 0.500 | 1.000 | 1.000 | 0.025 |
| speech | 9 | 0.000 | 0.222 | 0.222 | 0.000 |
| visual | 11 | 0.091 | 0.091 | 0.364 | 0.004 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 7 | 0.000 | 0.000 | 0.000 |
| indoor1 | 4 | 0.750 | 1.000 | 0.049 |
| indoor2 | 4 | 0.250 | 0.500 | 0.025 |
| lecture1 | 8 | 0.000 | 0.000 | 0.000 |
| lecture2 | 7 | 0.143 | 0.429 | 0.005 |
| screen-recording | 8 | 0.000 | 0.000 | 0.000 |
| streetcctv1 | 6 | 0.667 | 0.833 | 0.048 |
| streetcctv2 | 6 | 0.167 | 0.500 | 0.017 |
