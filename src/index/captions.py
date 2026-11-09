"""Layer C: frame captions, dedupe, and the caption index.

Captions are generated on sparse keyframes (every ``captions.keyframe_interval_s``
seconds, capped per video), cached in ``data/captions/<video_id>.json``, and
indexed with the same bge-small + BM25 stack as the other text layers.

The default backend is BLIP-base because it captions a keyframe in ~0.4 s on
the 3060. Qwen2.5-VL-3B is supported via ``captions.backend: qwen`` and is the
model used by the optional top-3 reranker.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.config import Config
from src.index import store
from src.index.frames import resolve_device
from src.index.text import (
    DEFAULT_TEXT_MODEL,
    encode_texts,
    load_text_encoder,
    normalize_text,
    text_similarity,
)
from src.utils.logging import StageTimer, get_logger

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in the core requirements
    tqdm = None  # type: ignore[assignment]

_LOG = get_logger("index.captions")

SCHEMA_VERSION = 1
DEFAULT_BACKEND = "blip"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_INTERVAL_S = 5.0
DEFAULT_MAX_KEYFRAMES = 300
DEFAULT_DEDUPE_SIMILARITY = 0.9
DEFAULT_MAX_GAP_S = 10.0

CAPTION_PROMPT = (
    "Describe the action, the people, and the objects in this frame "
    "in one detailed sentence."
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def caption_path(config: Config, video_id: str) -> Path:
    return config.captions_dir / f"{video_id}.json"


def caption_index_paths(config: Config, video_id: str) -> tuple[Path, Path]:
    base = config.indexes_dir / video_id
    return base / "caption.faiss", base / "caption.ids.json"


def _read_metadata(config: Config, video_id: str) -> dict[str, Any]:
    metadata_path = config.frames_dir / video_id / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"frame metadata not found: {metadata_path}. "
            f"Run scripts/index_video.py --video <video> first."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not payload.get("frames"):
        raise ValueError(f"no frames listed in {metadata_path}")
    return payload


# ---------------------------------------------------------------------------
# Keyframes and model
# ---------------------------------------------------------------------------


def select_caption_keyframes(
    frames: list[dict[str, Any]],
    config: Config,
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Pick one frame every ``captions.keyframe_interval_s`` seconds."""
    interval = float(config.get("captions.keyframe_interval_s", DEFAULT_INTERVAL_S))
    max_keyframes = int(config.get("captions.max_keyframes", DEFAULT_MAX_KEYFRAMES) or 0)
    if interval <= 0:
        raise ValueError("captions.keyframe_interval_s must be positive")

    selected: list[dict[str, Any]] = []
    last: Optional[float] = None
    for frame in frames:
        timestamp = float(frame["timestamp_s"])
        if last is None or timestamp - last >= interval:
            selected.append(frame)
            last = timestamp
            if max_keyframes and len(selected) >= max_keyframes:
                break
    return selected


@lru_cache(maxsize=2)
def load_caption_model(
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_CAPTION_MODEL,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Load BLIP or Qwen2.5-VL once per process."""
    import torch

    resolved = resolve_device(device_preference)
    backend = backend.lower()
    if backend == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_name = model_name or DEFAULT_QWEN_MODEL
        _LOG.info("loading caption model %s on %s (qwen)", model_name, resolved)
        processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.bfloat16 if resolved == "cuda" else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=resolved,
        )
        model.eval()
        return {
            "backend": "qwen",
            "model_name": model_name,
            "processor": processor,
            "model": model,
            "device": resolved,
            "dtype": dtype,
        }

    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_name = model_name or DEFAULT_CAPTION_MODEL
    _LOG.info("loading caption model %s on %s (blip)", model_name, resolved)
    processor = BlipProcessor.from_pretrained(model_name)
    dtype = torch.float16 if resolved == "cuda" else torch.float32
    model = (
        BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        .to(resolved)
    )
    model.eval()
    return {
        "backend": "blip",
        "model_name": model_name,
        "processor": processor,
        "model": model,
        "device": resolved,
        "dtype": dtype,
    }


def caption_image(engine: dict[str, Any], frame_path: str | Path) -> str:
    """Caption one image."""
    import torch

    image = Image.open(frame_path).convert("RGB")
    processor = engine["processor"]
    model = engine["model"]
    device = engine["device"]
    dtype = engine["dtype"]

    if engine["backend"] == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=48, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    inputs = processor(images=image, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Dedupe into segments
# ---------------------------------------------------------------------------


def dedupe_captions(
    entries: list[dict[str, Any]],
    *,
    similarity_threshold: float = DEFAULT_DEDUPE_SIMILARITY,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    duration_s: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Merge consecutive keyframes whose captions are nearly identical."""
    segments: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for entry in sorted(entries, key=lambda item: float(item["timestamp_s"])):
        caption = str(entry.get("caption") or "").strip()
        if not caption:
            continue
        timestamp = float(entry["timestamp_s"])
        normalized = normalize_text(caption)

        if (
            current is not None
            and timestamp - current["last_timestamp"] <= max_gap_s
            and text_similarity(current["_normalized"], normalized)
            >= similarity_threshold
        ):
            current["end"] = timestamp
            current["last_timestamp"] = timestamp
            current["keyframe_count"] += 1
            current["keyframes"].append(
                {"timestamp_s": round(timestamp, 3), "path": entry["path"]}
            )
            if len(caption) > len(current["text"]):
                current["text"] = caption
                current["path"] = entry["path"]
                current["_normalized"] = normalized
            continue

        current = {
            "start": timestamp,
            "end": timestamp,
            "last_timestamp": timestamp,
            "text": caption,
            "path": entry["path"],
            "keyframe_count": 1,
            "keyframes": [
                {"timestamp_s": round(timestamp, 3), "path": entry["path"]}
            ],
            "_normalized": normalized,
        }
        segments.append(current)

    # Captions persist until the next keyframe, capped by max_gap_s.
    for position, segment in enumerate(segments):
        if position + 1 < len(segments):
            target = float(segments[position + 1]["start"])
        elif duration_s:
            target = float(duration_s)
        else:
            target = float(segment["end"])
        if target > segment["end"]:
            segment["end"] = min(target, float(segment["end"]) + max_gap_s)

    for index, segment in enumerate(segments):
        segment.pop("_normalized", None)
        segment.pop("last_timestamp", None)
        segment["index"] = index
    return segments


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def caption_settings(config: Config, embedding_model: str) -> dict[str, Any]:
    backend = str(config.get("captions.backend", DEFAULT_BACKEND) or DEFAULT_BACKEND)
    if backend.lower() == "qwen":
        model = str(
            config.get("captions.qwen_model") or DEFAULT_QWEN_MODEL
        )
    else:
        model = str(
            config.get("captions.model", DEFAULT_CAPTION_MODEL)
            or DEFAULT_CAPTION_MODEL
        )
    return {
        "backend": backend,
        "model": model,
        "keyframe_interval_s": float(
            config.get("captions.keyframe_interval_s", DEFAULT_INTERVAL_S)
        ),
        "max_keyframes": int(
            config.get("captions.max_keyframes", DEFAULT_MAX_KEYFRAMES) or 0
        ),
        "dedupe_similarity": float(
            config.get("captions.dedupe_similarity", DEFAULT_DEDUPE_SIMILARITY)
        ),
        "max_gap_s": float(config.get("captions.max_gap_s", DEFAULT_MAX_GAP_S)),
        "embedding_model": embedding_model,
    }


_CAPTION_EXTRACTION_KEYS = (
    "backend",
    "model",
    "keyframe_interval_s",
    "max_keyframes",
)


def _extraction_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: settings.get(key) for key in _CAPTION_EXTRACTION_KEYS}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _caption_cache_valid(
    payload: Optional[dict[str, Any]],
    metadata: dict[str, Any],
) -> bool:
    if not payload:
        return False
    if payload.get("source_hash") != metadata.get("source_hash"):
        return False
    return "captions" in payload


def _cached_index(
    ids_path: Path,
    index_path: Path,
    metadata: dict[str, Any],
    settings: dict[str, Any],
) -> Optional[dict[str, Any]]:
    payload = _load_json(ids_path)
    if not payload:
        return None
    if payload.get("settings") != settings:
        return None
    if payload.get("source_hash") != metadata.get("source_hash"):
        return None
    if payload.get("empty"):
        return payload
    if not index_path.is_file():
        return None
    if int(payload.get("count", -1)) != len(payload.get("records") or []):
        return None
    return payload


def _write_empty_sidecar(
    video_id: str,
    config: Config,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    index_path, ids_path = caption_index_paths(config, video_id)
    index_path.unlink(missing_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "layer": "caption",
        "backend": settings["backend"],
        "model": settings["model"],
        "embedding_model": settings["embedding_model"],
        "dim": None,
        "normalized": True,
        "count": 0,
        "empty": True,
        "reason": reason,
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "caption_path": store.relative_path(
            caption_path(config, video_id), config.repo_root
        ),
        "settings": settings,
        "stats": {"keyframes": 0, "segments": 0},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": [],
    }
    store.write_json(ids_path, payload)
    _LOG.warning("%s has no captions (%s); wrote an empty caption sidecar", video_id, reason)
    return payload


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def index_caption_video(
    video_id: str,
    config: Config,
    *,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Build (or reuse) the Layer C index for one video."""
    metadata = _read_metadata(config, video_id)
    embedding_model = str(
        config.get("text.embedding_model") or DEFAULT_TEXT_MODEL
    )
    batch_size = int(batch_size or config.get("text.batch_size", 32))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    settings = caption_settings(config, embedding_model)
    index_path, ids_path = caption_index_paths(config, video_id)

    if not overwrite:
        cached = _cached_index(ids_path, index_path, metadata, settings)
        if cached is not None:
            _LOG.info(
                "cache hit: %s caption index (%d segments, %s)",
                video_id,
                int(cached.get("count", 0)),
                embedding_model,
            )
            return cached, True
        if index_path.exists() or ids_path.exists():
            _LOG.info("caption index stale; rebuilding %s", video_id)

    caption_file = caption_path(config, video_id)
    extraction = _load_json(caption_file)
    extraction_cached = not overwrite and _caption_cache_valid(extraction, metadata)
    if extraction_cached and extraction is not None:
        stored = extraction.get("settings") or {}
        expected = _extraction_settings(settings)
        if any(
            key in stored and stored[key] != value
            for key, value in expected.items()
        ):
            _LOG.info(
                "reusing cached captions with different settings; "
                "pass --overwrite to re-caption"
            )

    if not extraction_cached:
        keyframes = select_caption_keyframes(
            metadata["frames"], config, show_progress=show_progress
        )
        if not keyframes:
            extraction = {
                "schema_version": SCHEMA_VERSION,
                "tool": "videoseek",
                "video_id": video_id,
                "source_path": metadata.get("source_path"),
                "source_hash": metadata.get("source_hash"),
                "settings": _extraction_settings(settings),
                "stats": {"keyframes": 0, "segments": 0},
                "captions": [],
                "segments": [],
                "created_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            store.write_json(caption_file, extraction)
            return (
                _write_empty_sidecar(
                    video_id, config, metadata, settings, reason="no_keyframes"
                ),
                False,
            )

        engine = load_caption_model(
            settings["backend"],
            settings["model"],
            str(config.get("captions.device", "auto")),
        )
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                total=len(keyframes), unit="frame", desc="caption", dynamic_ncols=True
            )
        extracted: list[dict[str, Any]] = []
        try:
            for keyframe in keyframes:
                started = time.perf_counter()
                caption = caption_image(
                    engine, config.resolve(keyframe["path"])
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                extracted.append(
                    {
                        "timestamp_s": float(keyframe["timestamp_s"]),
                        "path": keyframe["path"],
                        "caption": caption,
                        "latency_ms": round(latency_ms, 2),
                    }
                )
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        segments = dedupe_captions(
            extracted,
            similarity_threshold=float(settings["dedupe_similarity"]),
            max_gap_s=float(settings["max_gap_s"]),
            duration_s=float(metadata.get("duration_s") or 0.0) or None,
        )
        extraction = {
            "schema_version": SCHEMA_VERSION,
            "tool": "videoseek",
            "video_id": video_id,
            "source_path": metadata.get("source_path"),
            "source_hash": metadata.get("source_hash"),
            "backend": settings["backend"],
            "model": settings["model"],
            "settings": _extraction_settings(settings),
            "stats": {
                "keyframes": len(extracted),
                "segments": len(segments),
                "avg_latency_ms": round(
                    sum(item["latency_ms"] for item in extracted)
                    / max(1, len(extracted)),
                    2,
                ),
            },
            "captions": extracted,
            "segments": segments,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        store.write_json(caption_file, extraction)
    else:
        assert extraction is not None
        _LOG.info("cache hit: captions %s", caption_file)

    duration = float(metadata.get("duration_s") or 0.0) or None
    segments = dedupe_captions(
        (extraction or {}).get("captions") or [],
        similarity_threshold=float(settings["dedupe_similarity"]),
        max_gap_s=float(settings["max_gap_s"]),
        duration_s=duration,
    )
    if not segments:
        return (
            _write_empty_sidecar(
                video_id, config, metadata, settings, reason="no_captions"
            ),
            False,
        )

    text_device = resolve_device(config.get("text.device", "auto"))
    encoder = load_text_encoder(embedding_model, text_device)
    with StageTimer("embed captions", _LOG):
        vectors = encode_texts(
            [segment["text"] for segment in segments],
            encoder,
            batch_size=batch_size,
        )
    if vectors.shape[0] != len(segments):
        raise RuntimeError(
            f"caption embedding count {vectors.shape[0]} does not match segments {len(segments)}"
        )

    with StageTimer("faiss index", _LOG):
        index = store.build_flat_ip(vectors)
        store.save_faiss(index, index_path)

    records = []
    for position, segment in enumerate(segments):
        records.append(
            {
                "doc_id": f"{video_id}:caption:{int(segment['index']):04d}",
                "video_id": video_id,
                "layer": "caption",
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "path": segment["path"],
                "text": segment["text"],
                "meta": {
                    "segment_idx": int(segment["index"]),
                    "keyframe_count": int(segment.get("keyframe_count", 1)),
                    "keyframes": segment.get("keyframes") or [],
                    "position": position,
                },
            }
        )

    stats = (extraction or {}).get("stats") or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "layer": "caption",
        "backend": settings["backend"],
        "model": settings["model"],
        "embedding_model": embedding_model,
        "device": text_device,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "count": len(records),
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "caption_path": store.relative_path(caption_file, config.repo_root),
        "settings": settings,
        "stats": {
            "keyframes": stats.get("keyframes", 0),
            "segments": len(segments),
            "avg_latency_ms": stats.get("avg_latency_ms"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": records,
    }
    store.write_json(ids_path, payload)
    _LOG.info(
        "wrote %d caption segments (%d-d) from %d keyframes to %s",
        len(records),
        payload["dim"],
        stats.get("keyframes", 0),
        store.relative_path(index_path, config.repo_root),
    )
    return payload, False
