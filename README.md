# VideoSeek

Search long videos with plain English. Ask a question, get a timestamp, a thumbnail, and a playable clip.

> **Status: P4 complete — segments and fusion.** Results are returned as timestamped segments with source badges. On the 50-query dev set, A+B scores Recall@5 **0.880** vs **0.860** for frames alone; on the 20-query held-out set both score **0.600**. Layer A remains the strongest standalone layer, and the object layer's earlier gain did not generalize. Speech, OCR, captions, and UI are next.

---

## What it will do

Index a video once, then search it across five layers:

| Layer | What it indexes | Example query |
|---|---|---|
| Frames | scene and place | "empty road at night" |
| Objects | people, cars, bags | "person in a red jacket" |
| Speech | what is said | "where he says gradient descent" |
| On-screen text | slides, signs, labels | "the slide about transformers" |
| Captions | actions and context | "someone arguing" |

Results are returned as time segments with a source badge, so you can see **why** each hit matched.

## Requirements

- **Python 3.10+**
- **FFmpeg** on `PATH` (required for audio and video decoding)
- **GPU recommended** (tested target: NVIDIA RTX 3060 12 GB). CPU works but indexing is much slower.
- **Disk:** model weights several GB on first run; roughly 50–400 MB per hour of indexed video depending on thumbnail settings.
- **RAM:** 8 GB minimum; 16 GB recommended.

## Install

```bash
git clone https://github.com/Fuad-Rafi/Natural-language-video-search-engine.git
cd Natural-language-video-search-engine

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Exact versions from the verified Windows + CUDA 13 environment are pinned in `requirements-lock.txt`.

Verify FFmpeg:

```bash
ffmpeg -version
```

## Usage

### Index a video (available)

```bash
# 1. Put a video in data/videos/ (or pass any path)
python scripts/index_video.py --video data/videos/sample.mp4

# 2. Verify frames and timestamps against the source
python scripts/verify_frames.py --video data/videos/sample.mp4
```

Frames are written to `data/frames/<video_id>/` (`000000.jpg`, `metadata.json`, `frames.jsonl`), with a per-run log in `logs/`.

### Search (available)

```bash
# 3. Build the Layer A index (SigLIP + FAISS; downloads ~400 MB on first run)
python scripts/index_frames.py --video data/videos/sample.mp4

# 4. Search it
python scripts/search.py --query "a close-up of a human eye" --video sample

# 5. Build the object layer and search crops
python scripts/index_objects.py --video data/videos/streetcctv1.mp4
python scripts/search.py --query "a red car parked on the street" --layer objects --video dashcam
```

Search returns frame timestamps with cosine scores. Omitting `--video` searches every indexed video. The first run loads the SigLIP model (about 2 seconds on the RTX 3060).

### Evaluate (available)

```bash
python scripts/evaluate.py --mode A    # frames
python scripts/evaluate.py --mode B    # objects
python scripts/evaluate.py --mode AB   # RRF fusion, segments

# Held-out validation
python scripts/evaluate.py --mode AB --queries benchmarks/heldout.csv
```

All modes merge hits into segments (gap 3 s, minimum 1 s) and report Recall@1/5/10, segment mIoU, duplicate rate, and latency.

### UI (planned, P8)

```bash
python app.py
```

Indexing is cached at every stage: re-running the same video reuses existing frames and embeddings.

## Results

Segment policy: every config merges hits with a 3-second gap and a 1-second minimum before scoring, so Recall and mIoU are comparable across A, B, and A+B.

| Config | Dev Recall@5 (50) | Held-out Recall@5 (20) | Dev mIoU | Duplicate rate (raw → merged) |
|---|---|---|---|---|
| A — frames | 0.860 | **0.600** | **0.257** | 0.458 → 0.000 |
| B — objects | 0.440 | 0.450 | 0.042 | 0.112 → 0.000 |
| A+B — RRF | **0.880** | **0.600** | 0.242 | 0.384 → 0.000 |

The 50-query set is the development benchmark used for tuning; the 20-query held-out set is the honest headline (`benchmarks/heldout.lock.json`). A+B improves the dev set by 2 points and the object+spatial subset from 0.909 to **1.000**, but ties frames on held-out — so fusion is reported as parity, not a gain. Duplicate results fall to **0.000** after merging in every config.

Full tables, per-type breakdowns, and failure notes: `benchmarks/results/`, `benchmarks/failures/`.

## Configuration

Defaults live in `configs/default.yaml`:

```yaml
video:
  sample_fps: 1.0
  keyframe_fps: 0.5

search:
  top_k: 20
  rrf_k: 60
  rrf_window_s: 2.0
  merge_gap_s: 3.0
  min_segment_s: 1.0
  weights:
    frame: 1.0
    object: 1.0
```

Override per run with `--config path/to/config.yaml`.

## Models

Downloaded automatically on first use:

| Component | Model | Approx. size |
|---|---|---|
| Frame embeddings | SigLIP base patch16-224 | ~400 MB |
| Object detection | YOLO11n | ~6 MB |
| Speech | faster-whisper small | ~500 MB |
| OCR | PaddleOCR English | ~20 MB |
| Text embeddings | bge-small-en-v1.5 | ~130 MB |

Models are downloaded to the Hugging Face / Ultralytics cache, not into this repository.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Install FFmpeg and add it to `PATH` |
| CUDA out of memory | Use a smaller model, close other GPU apps, or run on CPU |
| No audio stream | The speech layer is skipped; other layers still work |
| No text found | OCR may need higher-resolution keyframes |
| Slow first run | Model download + one-time indexing |

## Data

No videos are included. Use your own footage or a public dataset that permits research use. Dataset files are not redistributed by this repository.

## License

**AGPL-3.0.** This project uses [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. See `LICENSE`.

If you need a permissive license for commercial use, replace the detector with an Apache-2.0/MIT model (for example RT-DETR via Transformers) or obtain a commercial Ultralytics license.

## Contributing

Issues and suggestions are welcome. Please open an issue before a large pull request.
