"""Layer A: SigLIP frame embeddings and the FAISS frame index.

Images are embedded in batches, L2-normalized, and stored in an exact
inner-product index. A JSON sidecar keeps the frame records aligned with the
FAISS positions, so search results always map back to a timestamp and a file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.index import store
from src.utils.logging import StageTimer, get_logger

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in the core requirements
    tqdm = None  # type: ignore[assignment]

_LOG = get_logger("index.frames")
DEFAULT_MODEL = "google/siglip-base-patch16-224"


def resolve_device(preference: Optional[str] = "auto") -> str:
    """Map ``auto|cuda|cpu`` to a usable torch device string."""
    value = (preference or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false; "
                "install a CUDA build of torch or pass --device cpu"
            )
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"unsupported device {preference!r}; use auto, cuda, or cpu")


@lru_cache(maxsize=2)
def load_siglip(model_name: str = DEFAULT_MODEL, device: str = "auto") -> tuple[Any, Any]:
    """Load the SigLIP processor and model once per process."""
    from transformers import AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    resolved = resolve_device(device)
    _LOG.info("loading %s on %s", model_name, resolved)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolved)
    model.eval()
    return processor, model


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def _pooled_features(output: Any) -> torch.Tensor:
    """Extract the embedding tensor from a transformers 4/5 output object."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise RuntimeError(
            f"model output {type(output).__name__} has no pooler_output; "
            "check the transformers version and SigLIP API"
        )
    return pooled


def embed_images(
    paths: Iterable[str | Path],
    processor: Any,
    model: Any,
    *,
    device: str = "auto",
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Return one normalized embedding per image, in input order."""
    image_paths = [Path(path) for path in paths]
    if not image_paths:
        raise ValueError("no image paths to embed")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    resolved = resolve_device(device)
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(image_paths), unit="img", desc="embed", dynamic_ncols=True)

    chunks: list[np.ndarray] = []
    try:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"frame not found: {path}")
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(resolved) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _pooled_features(model.get_image_features(**inputs))
            normalized = _l2_normalize(features)
            chunks.append(normalized.cpu().numpy().astype(np.float32))

            if progress is not None:
                progress.update(len(batch_paths))
    finally:
        if progress is not None:
            progress.close()

    if not chunks:
        raise RuntimeError("no embeddings were produced")
    return np.concatenate(chunks, axis=0)


def embed_texts(
    texts: Iterable[str],
    processor: Any,
    model: Any,
    *,
    device: str = "auto",
) -> np.ndarray:
    """Embed text queries with SigLIP's required max-length padding."""
    values = [str(text) for text in texts]
    if not values:
        raise ValueError("no texts to embed")

    resolved = resolve_device(device)
    inputs = processor(
        text=values,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(resolved) for key, value in inputs.items()}
    with torch.inference_mode():
        features = _pooled_features(model.get_text_features(**inputs))
    return _l2_normalize(features).cpu().numpy().astype(np.float32)


def frame_index_paths(config: Config, video_id: str) -> tuple[Path, Path]:
    base = config.indexes_dir / video_id
    return base / "frame.faiss", base / "frame.ids.json"


def _read_metadata(video_id: str, config: Config) -> dict[str, Any]:
    metadata_path = config.frames_dir / video_id / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"frame metadata not found: {metadata_path}. "
            f"Run scripts/index_video.py --video <video> first."
        )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid frame metadata: {metadata_path}: {exc}") from exc
    if not payload.get("frames"):
        raise ValueError(f"no frames listed in {metadata_path}")
    return payload


def _cached_index(
    ids_path: Path,
    index_path: Path,
    metadata: dict[str, Any],
    model_name: str,
) -> Optional[dict[str, Any]]:
    if not index_path.is_file() or not ids_path.is_file():
        return None
    try:
        payload = store.read_json(ids_path)
    except store.StoreError:
        return None
    if payload.get("model") != model_name:
        return None
    if payload.get("source_hash") != metadata.get("source_hash"):
        return None
    if payload.get("count") != len(metadata.get("frames") or []):
        return None
    return payload


def index_frame_video(
    video_id: str,
    config: Config,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Build (or reuse) the Layer A index for one video.

    Returns ``(sidecar_payload, reused_cache)``.
    """
    metadata = _read_metadata(video_id, config)
    frames: list[dict[str, Any]] = metadata["frames"]
    model_name = model_name or str(config.get("frames.model") or DEFAULT_MODEL)
    batch_size = int(batch_size or config.get("frames.batch_size", 32))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    index_path, ids_path = frame_index_paths(config, video_id)
    if not overwrite:
        cached = _cached_index(ids_path, index_path, metadata, model_name)
        if cached is not None:
            _LOG.info(
                "cache hit: %s frame index (%d vectors, %s)",
                video_id,
                int(cached.get("count", 0)),
                model_name,
            )
            return cached, True
        if index_path.exists() or ids_path.exists():
            _LOG.info("frame index stale; rebuilding %s", video_id)

    resolved_device = resolve_device(device or config.get("frames.device", "auto"))
    processor, model = load_siglip(model_name, resolved_device)

    image_paths = [config.resolve(frame["path"]) for frame in frames]
    with StageTimer("embed frames", _LOG):
        vectors = embed_images(
            image_paths,
            processor,
            model,
            device=resolved_device,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    if vectors.shape[0] != len(frames):
        raise RuntimeError(
            f"embedding count {vectors.shape[0]} does not match frames {len(frames)}"
        )

    with StageTimer("faiss index", _LOG):
        index = store.build_flat_ip(vectors)
        store.save_faiss(index, index_path)

    records = []
    for position, frame in enumerate(frames):
        timestamp = float(frame["timestamp_s"])
        records.append(
            {
                "doc_id": f"{video_id}:frame:{int(frame['index']):06d}",
                "video_id": video_id,
                "layer": "frame",
                "start": timestamp,
                "end": timestamp,
                "path": frame["path"],
                "text": None,
                "meta": {
                    "frame_idx": int(frame["index"]),
                    "source_frame_idx": int(frame["source_frame_index"]),
                    "timestamp_s": timestamp,
                    "position": position,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "layer": "frame",
        "model": model_name,
        "device": resolved_device,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "count": len(records),
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "sample_fps": metadata.get("sample_fps"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": records,
    }
    store.write_json(ids_path, payload)
    _LOG.info(
        "wrote %d vectors (%d-d) to %s",
        len(records),
        payload["dim"],
        store.relative_path(index_path, config.repo_root),
    )
    return payload, False
