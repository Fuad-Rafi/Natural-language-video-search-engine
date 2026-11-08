# VideoSeek

Search long videos with plain English. Ask a question, get a timestamp, a thumbnail, and a playable clip.

> **Status: P5 complete — speech.** Results are returned as timestamped segments with source badges. On the 50-query dev set, A+B+D reaches Recall@5 **0.960** vs **0.880** for A+B and **0.860** for frames alone; speech queries return the right moment in the top 3 for **9/9** dev queries. On the 20-query held-out set, A+B+D reaches **0.700** vs **0.600** for both frames and A+B, with speech hitting the top 3 for **3/4** held-out queries. OCR, captions, and UI are next.

---

## What it does

Index a video once, then search it across five layers:

| Layer | What it indexes | Example query | Status |
|---|---|---|---|
| Frames | scene and place | "empty road at night" | ✅ |
| Objects | people, cars, bags | "person in a red jacket" | ✅ |
| Speech | what is said | "where he says gradient descent" | ✅ |
| On-screen text | slides, signs, labels | "the slide about transformers" | planned (P6) |
| Captions | actions and context | "someone arguing" | planned (P7) |

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

# 6. Build the speech layer and search the transcript
python scripts/index_speech.py --video data/videos/lecture1.mp4
python scripts/search.py --query "where he says gradient descent" --layer speech --video lecture1
```

Search returns frame, object, or transcript hits; the evaluator merges them into segments. Omitting `--video` searches every indexed video. The first run loads the SigLIP model (about 2 seconds on the RTX 3060) and, for speech, the Whisper and BGE models.

### Evaluate (available)

```bash
python scripts/evaluate.py --mode A    # frames
python scripts/evaluate.py --mode B    # objects
python scripts/evaluate.py --mode D    # speech (transcript only)
python scripts/evaluate.py --mode AB   # RRF fusion, segments
python scripts/evaluate.py --mode ABD  # frames + objects + speech

# Speech queries only, and held-out validation
python scripts/evaluate.py --mode D --types speech
python scripts/evaluate.py --mode D --types speech --queries benchmarks/heldout.csv
python scripts/evaluate.py --mode ABD --queries benchmarks/heldout.csv
```

All modes merge hits into segments (gap 3 s, minimum 1 s) and report Recall@1/3/5/10, segment mIoU, duplicate rate, and latency. Speech queries are routed to Layer D inside A+B+D: frame hits saturate the transcript timeline, so naive RRF lets a visual cluster outrank the correct spoken passage (see `documentation/CHALLENGES.md`, C-27).

### UI (planned, P8)

```bash
python app.py
```

Indexing is cached at every stage: re-running the same video reuses existing frames and embeddings.

## Results

Segment policy: every config merges hits with a 3-second gap and a 1-second minimum before scoring, so Recall and mIoU are comparable across A, B, D, A+B, and A+B+D.

| Config | Dev Recall@5 | Held-out Recall@5 | Dev mIoU | Duplicate rate (raw → merged) |
|---|---|---|---|---|
| A — frames | 0.860 (50) | **0.600** (20) | **0.257** | 0.458 → 0.000 |
| B — objects | 0.440 (50) | 0.450 (20) | 0.042 | 0.112 → 0.000 |
| D — speech | **1.000** (9) | **1.000** (4) | **0.310** | 0.000 → 0.000 |
| A+B — RRF | **0.880** (50) | **0.600** (20) | 0.242 | 0.384 → 0.000 |
| A+B+D — routed | **0.960** (50) | **0.700** (20) | 0.280 | 0.328 → 0.000 |

The 50-query set is the development benchmark used for tuning; the 20-query held-out set is the honest headline (`benchmarks/heldout.lock.json`). Parentheses show query counts: the D row is scored only on speech queries (9 dev, 4 held-out), while the other rows cover the full set. A+B improves dev by 2 points but ties frames on held-out. Adding speech and routing speech queries to Layer D raises held-out Recall@5 from 0.600 to **0.700** and mIoU from 0.115 to **0.163** — the first held-out gain beyond frames. On dev, speech alone returns the correct moment in the top 3 for **9/9** queries and the top 5 for **9/9**; held-out is **3/4** top-3 and **4/4** top-5. Duplicate results fall to **0.000** after merging in every config.

Full tables, per-type breakdowns, and failure notes: `benchmarks/results/`, `benchmarks/failures/`.

## Configuration

Defaults live in `configs/default.yaml`:

```yaml
video:
  sample_fps: 1.0
  keyframe_fps: 0.5

speech:
  model: small
  chunk_seconds: 30
  overlap_seconds: 5

text:
  embedding_model: BAAI/bge-small-en-v1.5
  use_bm25: true
  query_prefix: true

search:
  top_k: 20
  rrf_k: 60
  rrf_window_s: 2.0
  merge_gap_s: 3.0
  min_segment_s: 1.0
  route_speech: true
  weights:
    frame: 1.0
    object: 1.0
    speech: 1.2
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
| Whisper CUDA DLL error | Install `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` (already in the lock file), or run `--device cpu` |
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
