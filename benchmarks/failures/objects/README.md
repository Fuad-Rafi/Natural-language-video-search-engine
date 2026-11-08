# Layer B object failure notes

Segment policy: gap ≤ 3 s, minimum 1 s.

| Set | Queries | R@1 | R@5 | R@10 | mIoU |
|---|---|---|---|---|---|
| Dev — all | 50 | 0.220 | 0.440 | 0.520 | 0.042 |
| Dev — object+spatial | 11 | 0.455 | **0.818** | 0.818 | 0.078 |
| Held-out — all | 20 | 0.150 | 0.450 | 0.550 | 0.042 |

Object-only dev R@5 is 0.778, below Layer A's 0.889. The layer is retained for
fusion and badges, but its standalone gain is **not confirmed** (P3 is 🟡).

## Object-only misses

| Query | Predicted | Ground truth | Cause |
|---|---|---|---|
| q045 — a white car driving ahead at night | 108 s | 85–100 s | Several valid night-car moments exist; taillights make white cars classify as red. |
| q048 — a red car parked on the street | 228 s | 255–270 s | Multiple red cars are parked/stopped along the drive; the annotated instance ranks lower. |

Both are single-interval annotation limits, not missing detections. The target
crops exist (verified visually).

## Non-COCO queries

`a red umbrella` and `a record player with a red vinyl record` are outside the
COCO class list used for crops. Layer B cannot represent them directly; both
still hit through scene context in the frame layer.

## What remains useful

- Object+spatial dev R@5 reaches 0.818 (Layer A: 0.909; A+B: 1.000).
- A+B recovers the distant red car that frames miss, and keeps Layer A's other
  object hits.
- The index contains 5,394 crops from 4,036 tracks (12 s segments plus 2 s anchors).

`gallery.jpg` is local and gitignored.
