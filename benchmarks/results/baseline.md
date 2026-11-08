# Baseline results — Layer A (frames)

- Generated: 2026-11-08T13:37:15+00:00
- Model: `google/siglip-base-patch16-224`
- Queries: 50 across 8 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.480 |
| Recall@5 | 0.740 |
| Recall@10 | 0.820 |
| mIoU (1 s window) | 0.032 |
| Latency p50 | 9 ms |
| Latency p95 | 17 ms |

## Per query type

| Type | Queries | R@1 | R@5 | R@10 | mIoU |
|---|---|---|---|---|---|
| action | 5 | 0.600 | 1.000 | 1.000 | 0.047 |
| caption | 1 | 0.000 | 1.000 | 1.000 | 0.000 |
| counting | 1 | 1.000 | 1.000 | 1.000 | 0.111 |
| object | 9 | 0.444 | 0.778 | 0.889 | 0.036 |
| ocr | 12 | 0.667 | 0.833 | 0.833 | 0.044 |
| spatial | 2 | 0.500 | 0.500 | 0.500 | 0.025 |
| speech | 9 | 0.333 | 0.444 | 0.667 | 0.016 |
| visual | 11 | 0.364 | 0.727 | 0.818 | 0.020 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 7 | 0.143 | 0.714 | 0.010 |
| indoor1 | 4 | 0.750 | 0.750 | 0.049 |
| indoor2 | 4 | 0.750 | 1.000 | 0.061 |
| lecture1 | 8 | 0.500 | 0.625 | 0.035 |
| lecture2 | 7 | 0.429 | 0.571 | 0.018 |
| screen-recording | 8 | 0.625 | 0.875 | 0.042 |
| streetcctv1 | 6 | 0.500 | 0.833 | 0.042 |
| streetcctv2 | 6 | 0.333 | 0.667 | 0.019 |
