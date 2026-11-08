# P2 custom benchmark

Frozen on **2026-11-08**. Do not edit `queries.csv` after seeing results.

- **Queries:** 50
- **Videos:** 8 (~74.6 minutes, 4,479 sampled frames at 1 fps)
- **Query file SHA-256:** `0b728db337406e3195e9063323eaf0938ac241e5e96ab499bfa377c9885f62bd`

## Composition

| Video | Content type | Length | Frames | Queries |
|---|---|---|---|---|
| `lecture1` | Lecture / slides (IELTS writing) | 19:59 | 1,198 | 8 |
| `lecture2` | Lecture / blackboard (6.006 algorithms) | 15:20 | 920 | 7 |
| `streetcctv1` | Street / CCTV (rainy crossing) | 4:02 | 242 | 6 |
| `streetcctv2` | Street / CCTV (Abbey Road) | 7:02 | 423 | 6 |
| `indoor1` | Indoor (coffee / lifestyle) | 7:51 | 472 | 4 |
| `indoor2` | Indoor (family / gaming) | 4:31 | 271 | 4 |
| `screen-recording` | Screen recording (Windows repair) | 5:42 | 343 | 8 |
| `dashcam` | Dashcam (Vancouver day/night) | 10:10 | 610 | 7 |

## Query types

| Type | Queries | Expected winning layer |
|---|---|---|
| speech | 9 | D — Whisper (P5) |
| ocr | 12 | E — PaddleOCR (P6) |
| object | 9 | B — YOLO + crops (P3) |
| visual | 11 | A — SigLIP frames (current) |
| spatial | 2 | B + geometry |
| action | 5 | A/C — weak until captions (P7) |
| caption | 1 | C — VLM captions (P7) |
| counting | 1 | B + caption (weak) |

## Baseline — Layer A (frames only)

| Metric | Value |
|---|---|
| Recall@1 | 0.480 |
| Recall@5 | **0.740** |
| Recall@10 | 0.820 |
| mIoU (1 s window) | **0.032** |
| Latency p50 | 9 ms |
| Latency p95 | 17 ms |

mIoU is low by construction: Layer A returns a single frame, evaluated as a 1-second window. Temporal merging arrives in P4.

### Per query type

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

Full artifacts: `results/baseline.json`, `results/baseline.md`, `failures/`.

## Layer B — objects (P3)

| Metric | Value |
|---|---|
| Recall@1 | 0.200 |
| Recall@5 | 0.340 |
| Recall@10 | 0.480 |
| mIoU (1 s window) | 0.014 |
| Latency p50 | 9 ms |
| Latency p95 | 11 ms |

Overall B is low by design: an object index cannot answer speech, OCR, or scene-level queries. On the **object + spatial subset (11 queries)** it is the relevant layer:

| Layer | Recall@5 (object+spatial) | mIoU |
|---|---|---|
| A — frames | 0.727 | 0.034 |
| B — objects | **0.818** | 0.037 |

Spatial queries improve from 0.500 to **1.000**; object-only Recall@5 ties at 0.778 with mIoU 0.036 → 0.040.

Object crops use YOLO11n + ByteTrack with 12 s segments, 2 s detection anchors, and SigLIP crop embeddings. Query-side class/color/night constraints are applied before ranking.

## Reproduce

```bash
python scripts/index_video.py --video data/videos/<file>.mp4
python scripts/index_frames.py --video data/videos/<file>.mp4
python scripts/index_objects.py --video data/videos/<file>.mp4
python scripts/evaluate.py                  # Layer A
python scripts/evaluate.py --layer objects  # Layer B
```

Videos, frames, and crops stay out of git. Only the CSV, results JSON/Markdown, and failure notes are versioned.
