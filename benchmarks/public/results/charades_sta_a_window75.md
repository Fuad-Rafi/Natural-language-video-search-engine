# Charades-STA test subset — Layer A

- Generated: 2026-11-09T07:18:49+00:00
- Model: `google/siglip-base-patch16-224`
- Videos: 50 · Queries: 399
- Proposal: 7.5s window centred on the top-1 frame
- Queries SHA-256: `ad9d2079d2cc4882…`

| Metric | Value |
|---|---|
| R@1 IoU 0.5 | 0.421 |
| R@1 IoU 0.7 | 0.241 |
| mIoU | 0.377 |
| Any overlap | 0.692 |

## Window sensitivity

| Window (s) | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---|---|---|
| 4.0 | 0.221 | 0.055 | 0.264 |
| 5.0 | 0.308 | 0.130 | 0.309 |
| 6.0 | 0.373 | 0.160 | 0.343 |
| 7.0 | 0.408 | 0.215 | 0.369 |
| 7.5 | 0.421 | 0.241 | 0.377 |
| 8.0 | 0.444 | 0.243 | 0.383 |
| 10.0 | 0.459 | 0.170 | 0.391 |
