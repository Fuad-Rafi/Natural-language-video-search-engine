# Layer A failure notes

Current policy: segments (gap ≤ 3 s, minimum 1 s), evaluated on the frozen sets.

| Set | Queries | R@1 | R@5 | R@10 | mIoU |
|---|---|---|---|---|---|
| Dev | 50 | 0.620 | 0.860 | 0.880 | 0.257 |
| Held-out | 20 | 0.550 | 0.600 | 0.700 | 0.137 |

19 of 50 dev queries miss at rank 1; the gallery shows up to 20. The held-out
gallery is written to `failures/heldout_a/`.

## Recurring causes

- **Speech and OCR queries** are answered only through visual context in Layer
  A; exact spoken phrases still fail there. Layer D now handles speech, and its
  failure gallery (`failures/speech/`) uses text tiles because transcripts have
  no frame thumbnail.
- **Repeated scenes** (roads, crossings, desktops) produce several valid
  moments, so a single annotated interval can mark a correct result as a miss.
- **Small objects** (distant red car, cyclist) are missed by whole-frame
  embeddings; this is the case for Layer B crops.
- **Action and counting** queries remain weak without captions.
- **OCR slide failures** often come from repeated titles (many slides say
  "essay 2 introduction") or handwritten chalkboards; unique screen/UI text
  ranks correctly (`failures/ocr/`).
- **Night scenes** are hard: taillights and low brightness dominate colour and
  object cues.

Point-era predictions are in git history. The current gallery is generated from
the segment evaluator; images are gitignored because they contain source frames.
