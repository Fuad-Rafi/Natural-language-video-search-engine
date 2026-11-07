# VideoSeek

Search long videos with plain English. Ask a question, get a timestamp, a thumbnail, and a playable clip.

> **Status: early scaffold.** The repository currently contains the public setup guide, requirements, and configuration. The indexing and search pipeline is being implemented phase by phase. The commands below describe the target interface and will be enabled as soon as the first pipeline version lands.

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

Verify FFmpeg:

```bash
ffmpeg -version
```

## Usage (planned interface)

```bash
# 1. Put a video in data/videos/ (or pass any path)
python scripts/index_video.py --video data/videos/sample.mp4

# 2. Search it
python scripts/search.py --query "where he says gradient descent"

# 3. Or use the web UI
python app.py
```

Indexing is cached: re-running the same video reuses existing frames, transcripts, and embeddings.

## Configuration

Defaults live in `configs/default.yaml`:

```yaml
video:
  sample_fps: 1.0
  keyframe_fps: 0.5

search:
  top_k: 20
  merge_gap_s: 3.0
  min_segment_s: 1.0
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
