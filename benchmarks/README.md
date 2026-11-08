# Custom benchmark and results

## Dev set (50 queries)

Frozen 2026-11-08. Used for development and ablation; **not** the headline.
Do not edit `queries.csv` after seeing results.

- **Queries:** 50
- **Videos:** 8 (~74.6 minutes, 4,479 sampled frames at 1 fps)
- **SHA-256:** `0b728db337406e3195e9063323eaf0938ac241e5e96ab499bfa377c9885f62bd`

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

## Held-out set (20 queries)

Frozen before P4 tuning; opened once after the system was final. This is the
headline result. Do not tune on it.

- **File:** `heldout.csv`
- **SHA-256:** `64b80d920204d73dae31c75341880a83c7f0b007d6dc27151c2b16da261b6868`
- **Manifest:** `heldout.lock.json`
- **Composition:** speech 4, OCR 4, object 5, visual 5, action 1, caption 1.

## Query types (dev)

| Type | Queries | Expected winning layer |
|---|---|---|
| speech | 9 | D — Whisper (done) |
| ocr | 12 | E — PaddleOCR (P6) |
| object | 9 | B — YOLO + crops (P3) |
| visual | 11 | A — SigLIP frames |
| spatial | 2 | B + geometry |
| action | 5 | A/C — weak until captions (P7) |
| caption | 1 | C — VLM captions (P7) |
| counting | 1 | B + caption (weak) |

## Evaluation policy

Every config is merged into segments with **gap ≤ 3 s** and **minimum 1 s**
before scoring, so A, B, and A+B are comparable.

- **Recall@k:** the k-th segment counts if it overlaps any ground-truth interval.
- **mIoU:** temporal IoU of the top-1 segment against the best GT interval.
- **Duplicate rate:** fraction of top-10 results within 3 s of an earlier result in the same video. Reported raw and after merging.

## Dev results — 50 queries

| Config | R@1 | R@3 | R@5 | R@10 | mIoU | Dup raw | Dup merged | p50 |
|---|---|---|---|---|---|---|---|---|
| A — frames | 0.620 | 0.760 | 0.860 | 0.880 | **0.257** | 0.458 | 0.000 | 8 ms |
| B — objects | 0.220 | 0.340 | 0.440 | 0.520 | 0.042 | 0.112 | 0.000 | 25 ms |
| D — speech (9) | 0.778 | **1.000** | **1.000** | **1.000** | **0.310** | 0.000 | 0.000 | 17 ms |
| A+B — RRF | 0.620 | 0.780 | 0.880 | 0.900 | 0.242 | 0.384 | 0.000 | 13 ms |
| A+B+D — routed | **0.700** | 0.860 | **0.960** | **0.960** | 0.280 | 0.328 | 0.000 | 12 ms |

Object/spatial subset (11 queries): A 0.909, B 0.818, A+B **1.000**.
Speech subset (9 queries): A+B 0.556 R@5, D **1.000** R@5, A+B+D (routed) **1.000** R@5.
A+B+D routes speech queries to Layer D; naive RRF scored only 0.556 R@5 on the speech subset because frame hits saturate every transcript chunk (C-27).
Per-type tables: `results/baseline.md`, `results/objects.md`, `results/speech.md`, `results/ab.md`, `results/abd.md`.

## Held-out results — 20 queries

| Config | R@1 | R@5 | R@10 | mIoU | Dup raw | Dup merged |
|---|---|---|---|---|---|---|
| A — frames | **0.550** | 0.600 | 0.700 | 0.137 | 0.450 | 0.000 |
| B — objects | 0.150 | 0.450 | 0.550 | 0.042 | 0.148 | 0.000 |
| D — speech (4) | **0.750** | **1.000** | **1.000** | **0.311** | 0.000 | 0.000 |
| A+B — RRF | 0.500 | 0.600 | 0.700 | 0.115 | 0.375 | 0.000 |
| A+B+D — routed | **0.550** | **0.700** | **0.750** | **0.163** | 0.265 | 0.000 |

A+B ties frames on held-out. Adding speech is the first config to beat frames: A+B+D raises Recall@5 from 0.600 to **0.700** and mIoU from 0.137 to **0.163**. The gain comes from the 4 speech queries, where the A+B speech subset scores 0.500 R@5 and Layer D scores **1.000**; routing speech queries to D is what generalizes. Speech top-3 is 3/4 on held-out and 9/9 on dev (the dev set contains 9 speech queries, not 10).

## Reproduce

```bash
python scripts/index_video.py --video data/videos/<file>.mp4
python scripts/index_frames.py --video data/videos/<file>.mp4
python scripts/index_objects.py --video data/videos/<file>.mp4
python scripts/index_speech.py --video data/videos/<file>.mp4

python scripts/evaluate.py --mode A
python scripts/evaluate.py --mode B
python scripts/evaluate.py --mode D --types speech
python scripts/evaluate.py --mode AB
python scripts/evaluate.py --mode ABD

# Held-out validation (run once, after tuning)
python scripts/evaluate.py --mode D --types speech --queries benchmarks/heldout.csv
python scripts/evaluate.py --mode ABD --queries benchmarks/heldout.csv
```

Videos, frames, and crops stay out of git. Only the CSVs, results JSON/Markdown, manifest, and failure notes are versioned.
