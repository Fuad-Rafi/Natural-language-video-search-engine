# Layer B object failure notes

Object-only results: Recall@1 0.200, Recall@5 0.340, mIoU 0.014.
On the Layer B subset (object + spatial, 11 queries): Recall@5 **0.818**, up from 0.727 for frames.

## Object-only misses

| Query | Predicted | Ground truth | Cause |
|---|---|---|---|
| q045 — a white car driving ahead at night | 108 s | 85–100 s | The night filter finds several valid “car ahead at night” moments (104–108 s) just outside the annotated window. Taillights make white cars classify as red, so color cannot disambiguate. |
| q048 — a red car parked on the street | 228 s | 255–270 s | Multiple red cars are parked/stopped along the drive. The object layer ranks the red cars at 228 s and 366 s above the annotated 260 s instance. |

Both are **single-interval annotation limits**, not missing detections: the target crops exist (verified visually) and the query is semantically satisfied at other timestamps.

## Non-COCO queries

- **q016 — a red umbrella** and **q030 — a record player with a red vinyl record** are outside the COCO class list used for crops. Layer B cannot represent them directly, though both happened to hit in the top 5 through scene context.
- Speech, OCR, and scene-level visual queries are not expected to work with an object-only index.

## What P3 changed

- Added YOLO11n + ByteTrack object crops, 12 s track segments, and 2 s detection anchors.
- Added SigLIP crop embeddings, a second FAISS index, and per-crop color/brightness metadata.
- Added query-side class/color/night constraints with graceful fallback.
- Fixed the white-van query (rank 1 at 240 s) and the red-car query (rank 2 at 190 s).

`gallery.jpg` is generated locally and gitignored because it contains source frames.
