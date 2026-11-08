# A+B (RRF) failure notes

Segment policy: gap ≤ 3 s, minimum 1 s; RRF `k=60`, frame/object weights 1.0,
2 s fusion window. Layer B is consulted only for object/spatial queries that
name a detectable class, and is skipped for `night` queries.

| Set | R@1 | R@5 | R@10 | mIoU | Dup raw → merged |
|---|---|---|---|---|---|
| Dev 50 | 0.620 | 0.880 | 0.900 | 0.242 | 0.384 → 0.000 |
| Held-out 20 | 0.500 | 0.600 | 0.700 | 0.115 | 0.375 → 0.000 |

## What happened

- On the dev set, fusion adds one object hit (the distant red car) and reaches
  **1.000** on the object+spatial subset, versus 0.909 for frames.
- On the held-out set it **ties frames at 0.600** and has lower mIoU. The dev
  gain did not generalize; the held-out set is small and Layer A is stronger.
- Fusion is therefore reported as **parity**, not an improvement, in the README.

## Why mIoU drops

B-only segments can be short (one crop) and locate the object less precisely
than a merged frame segment. Recall improves or ties, but top-1 temporal IoU
falls from 0.257 to 0.242 on dev and 0.137 to 0.115 on held-out.

## Duplicate behaviour

Raw duplicate rate is high for A (0.458) and falls to **0.000** after merging
for every config. The P4 duplicate target (<10% of top-10) is met.

`gallery.jpg` is local and gitignored.
