# Results — A+B+D+E+C (routed)

- Generated: 2026-11-09T07:08:16+00:00
- Model: `google/siglip-base-patch16-224`
- Mode: `ALL`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 50 across 8 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.580 |
| Recall@3 | 0.820 |
| Recall@5 | 0.920 |
| Recall@10 | 0.920 |
| mIoU (segments) | 0.148 |
| R@1 IoU 0.5 | 0.160 |
| R@1 IoU 0.7 | 0.020 |
| Duplicate rate (raw top-10) | 0.120 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 24 ms |
| Latency p95 | 139 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| action | 5 | 0.600 | 1.000 | 1.000 | 1.000 | 0.047 | 0.000 | 0.000 |
| caption | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| counting | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.111 | 0.000 | 0.000 |
| object | 9 | 0.556 | 0.778 | 1.000 | 1.000 | 0.147 | 0.178 | 0.000 |
| ocr | 12 | 0.500 | 0.667 | 0.750 | 0.750 | 0.038 | 0.000 | 0.000 |
| spatial | 2 | 0.500 | 1.000 | 1.000 | 1.000 | 0.025 | 0.000 | 0.000 |
| speech | 9 | 0.778 | 1.000 | 1.000 | 1.000 | 0.310 | 0.000 | 0.000 |
| visual | 11 | 0.545 | 0.727 | 0.909 | 0.909 | 0.221 | 0.400 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 7 | 0.286 | 1.000 | 0.182 |
| indoor1 | 4 | 0.750 | 1.000 | 0.245 |
| indoor2 | 4 | 0.750 | 1.000 | 0.178 |
| lecture1 | 8 | 0.500 | 0.875 | 0.183 |
| lecture2 | 7 | 0.714 | 0.857 | 0.212 |
| screen-recording | 8 | 0.625 | 0.875 | 0.088 |
| streetcctv1 | 6 | 0.667 | 1.000 | 0.079 |
| streetcctv2 | 6 | 0.500 | 0.833 | 0.052 |
