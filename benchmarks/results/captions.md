# Results — C (captions)

- Generated: 2026-11-09T07:06:04+00:00
- Model: `BAAI/bge-small-en-v1.5`
- Mode: `C`
- Temporal: gap 3.0 s, min 1.0 s
- Queries: 7 across 4 videos
- Query file SHA-256: `0b728db337406e31…`

| Metric | Value |
|---|---|
| Recall@1 | 0.286 |
| Recall@3 | 0.571 |
| Recall@5 | 0.714 |
| Recall@10 | 0.857 |
| mIoU (segments) | 0.157 |
| R@1 IoU 0.5 | 0.286 |
| R@1 IoU 0.7 | 0.000 |
| Duplicate rate (raw top-10) | 0.000 |
| Duplicate rate (merged) | 0.000 |
| Latency p50 | 20 ms |
| Latency p95 | 996 ms |

## Per query type

| Type | Queries | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|---|---|
| action | 5 | 0.200 | 0.600 | 0.800 | 1.000 | 0.100 | 0.000 | 0.000 |
| caption | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| counting | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.600 | 0.000 | 0.000 |

## Per video

| Video | Queries | R@1 | R@5 | mIoU |
|---|---|---|---|---|
| dashcam | 1 | 0.000 | 1.000 | 0.000 |
| indoor1 | 2 | 0.000 | 1.000 | 0.000 |
| indoor2 | 3 | 0.667 | 0.667 | 0.367 |
| streetcctv2 | 1 | 0.000 | 0.000 | 0.000 |
