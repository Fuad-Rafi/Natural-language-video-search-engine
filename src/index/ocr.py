"""Layer E: keyframe selection, PaddleOCR, dedupe, and the OCR index.

Slides and screens change slowly, so OCR runs on scene-change keyframes plus a
maximum interval, not every sampled frame. Consecutive keyframes whose text is
almost identical are merged into one time segment (the same slide shown for
30 seconds becomes one searchable result).

The raw OCR output is cached in ``data/ocr/<video_id>.json``; the FAISS index
can be rebuilt from that cache when the text embedding model changes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

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

_LOG = get_logger("index.ocr")

SCHEMA_VERSION = 1
DEFAULT_SCENE_THRESHOLD = 0.10
DEFAULT_MIN_INTERVAL_S = 1.0
DEFAULT_MAX_INTERVAL_S = 5.0
DEFAULT_MAX_GAP_S = 8.0
DEFAULT_DEDUPE_SIMILARITY = 0.9
DEFAULT_MIN_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def ocr_path(config: Config, video_id: str) -> Path:
    return config.ocr_dir / f"{video_id}.json"


def ocr_index_paths(config: Config, video_id: str) -> tuple[Path, Path]:
    base = config.indexes_dir / video_id
    return base / "ocr.faiss", base / "ocr.ids.json"


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
# Text helpers (unit-tested)
# ---------------------------------------------------------------------------


def normalize_ocr_text(text: str) -> str:
    """Lowercase and strip punctuation so OCR noise does not defeat dedupe."""
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------


def _thumbnail(path: str | Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def select_keyframes(
    frames: list[dict[str, Any]],
    config: Config,
    *,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Pick frames on scene change, with a maximum interval as a safety net."""
    min_interval = float(
        config.get("ocr.keyframe_min_interval_s", DEFAULT_MIN_INTERVAL_S)
    )
    max_interval = float(
        config.get("ocr.keyframe_max_interval_s", DEFAULT_MAX_INTERVAL_S)
    )
    threshold = float(config.get("ocr.scene_threshold", DEFAULT_SCENE_THRESHOLD))
    blank_std = float(config.get("ocr.blank_std_threshold", 0.01))
    max_keyframes = int(config.get("ocr.max_keyframes", 0) or 0)
    if min_interval < 0 or max_interval <= 0:
        raise ValueError("keyframe intervals must be positive")

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(frames), unit="frame", desc="keyframes", dynamic_ncols=True)

    selected: list[dict[str, Any]] = []
    previous: Optional[np.ndarray] = None
    last_kept: Optional[float] = None
    try:
        for frame in frames:
            path = config.resolve(frame["path"])
            thumbnail = _thumbnail(path)
            if progress is not None:
                progress.update(1)
            if thumbnail is None:
                continue

            timestamp = float(frame["timestamp_s"])
            std = float(thumbnail.std())
            if std < blank_std and selected:
                continue

            difference = (
                1.0 if previous is None else float(np.abs(thumbnail - previous).mean())
            )
            keep = (
                last_kept is None
                or (
                    timestamp - last_kept >= min_interval
                    and difference >= threshold
                )
                or timestamp - last_kept >= max_interval
            )
            if not keep:
                continue

            selected.append(
                {
                    **frame,
                    "scene_diff": round(difference, 4),
                    "thumbnail_std": round(std, 4),
                }
            )
            previous = thumbnail
            last_kept = timestamp
            if max_keyframes and len(selected) >= max_keyframes:
                break
    finally:
        if progress is not None:
            progress.close()
    return selected


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------


def _prepare_paddle_environment() -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def load_ocr_engine(config: Config) -> Any:
    """Load PaddleOCR.

    On Windows, importing PaddleOCR before torch can trigger a DLL load failure
    (paddlex imports modelscope, which imports torch). Import torch first.
    """
    _prepare_paddle_environment()
    import torch  # noqa: F401  (load before paddleocr; challenge C-29)

    from paddleocr import PaddleOCR

    lang = str(config.get("ocr.lang", "en") or "en")
    kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if not bool(config.get("ocr.enable_mkldnn", False)):
        kwargs["enable_mkldnn"] = False
    for key in (
        "text_detection_model_name",
        "text_recognition_model_name",
        "text_det_limit_side_len",
        "text_det_limit_type",
        "text_recognition_batch_size",
    ):
        value = config.get(f"ocr.{key}")
        if value is not None:
            kwargs[key] = value
    if not any(
        kwargs.get(name)
        for name in (
            "text_detection_model_name",
            "text_recognition_model_name",
        )
    ):
        kwargs["lang"] = lang
    _LOG.info("loading PaddleOCR (lang=%s)", lang)
    return PaddleOCR(**kwargs)


def _is_latin_text(text: str) -> bool:
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return False
    ascii_count = sum(1 for character in characters if character.isascii())
    return ascii_count / len(characters) >= 0.7


def read_frame_lines(
    engine: Any,
    frame_path: str | Path,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    allow_non_ascii: bool = False,
) -> list[dict[str, Any]]:
    """Run OCR on one image and return filtered text lines."""
    results = engine.predict(str(frame_path))
    if not results:
        return []
    result = results[0]
    texts = result.get("rec_texts") or []
    scores = result.get("rec_scores") or []
    boxes = result.get("rec_boxes")

    lines: list[dict[str, Any]] = []
    for index, (text, score) in enumerate(zip(texts, scores)):
        text = str(text).strip()
        confidence = float(score)
        if not text or confidence < min_confidence:
            continue
        if not allow_non_ascii and not _is_latin_text(text):
            continue
        box = None
        if boxes is not None:
            try:
                values = np.asarray(boxes[index]).reshape(-1)[:4]
                if values.size == 4:
                    box = [round(float(value), 1) for value in values]
            except (IndexError, TypeError, ValueError):
                box = None
        lines.append(
            {
                "text": text,
                "confidence": round(confidence, 4),
                "box": box,
            }
        )
    return lines


# ---------------------------------------------------------------------------
# Dedupe into segments
# ---------------------------------------------------------------------------


def dedupe_ocr_keyframes(
    keyframes: list[dict[str, Any]],
    *,
    similarity_threshold: float = DEFAULT_DEDUPE_SIMILARITY,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    duration_s: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Merge consecutive keyframes whose OCR text is nearly identical."""
    segments: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for keyframe in sorted(keyframes, key=lambda item: float(item["timestamp_s"])):
        lines = keyframe.get("lines") or []
        if not lines:
            continue
        text = " ".join(str(line["text"]) for line in lines).strip()
        normalized = normalize_ocr_text(text)
        if not normalized:
            continue
        timestamp = float(keyframe["timestamp_s"])
        confidence = max(float(line.get("confidence", 0.0)) for line in lines)

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
                {
                    "timestamp_s": round(timestamp, 3),
                    "path": keyframe["path"],
                    "scene_diff": keyframe.get("scene_diff"),
                }
            )
            if confidence > current["confidence"]:
                current.update(
                    {
                        "text": text,
                        "lines": lines,
                        "path": keyframe["path"],
                        "confidence": round(confidence, 4),
                        "_normalized": normalized,
                    }
                )
            continue

        current = {
            "start": timestamp,
            "end": timestamp,
            "last_timestamp": timestamp,
            "text": text,
            "path": keyframe["path"],
            "confidence": round(confidence, 4),
            "lines": lines,
            "keyframe_count": 1,
            "keyframes": [
                {
                    "timestamp_s": round(timestamp, 3),
                    "path": keyframe["path"],
                    "scene_diff": keyframe.get("scene_diff"),
                }
            ],
            "_normalized": normalized,
        }
        segments.append(current)

    # Text seen at a keyframe persists until the next keyframe. Extending the
    # segment end keeps long-lived slides searchable and gives frames+OCR
    # fusion a real temporal window. Blank stretches are capped by max_gap_s.
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


def ocr_settings(config: Config, embedding_model: str) -> dict[str, Any]:
    return {
        "lang": str(config.get("ocr.lang", "en") or "en"),
        "min_confidence": float(
            config.get("ocr.min_confidence", DEFAULT_MIN_CONFIDENCE)
        ),
        "enable_mkldnn": bool(config.get("ocr.enable_mkldnn", False)),
        "allow_non_ascii": bool(config.get("ocr.allow_non_ascii", False)),
        "text_detection_model_name": config.get("ocr.text_detection_model_name"),
        "text_recognition_model_name": config.get("ocr.text_recognition_model_name"),
        "text_det_limit_side_len": config.get("ocr.text_det_limit_side_len"),
        "text_det_limit_type": config.get("ocr.text_det_limit_type"),
        "text_recognition_batch_size": config.get("ocr.text_recognition_batch_size"),
        "keyframe_min_interval_s": float(
            config.get("ocr.keyframe_min_interval_s", DEFAULT_MIN_INTERVAL_S)
        ),
        "keyframe_max_interval_s": float(
            config.get("ocr.keyframe_max_interval_s", DEFAULT_MAX_INTERVAL_S)
        ),
        "scene_threshold": float(
            config.get("ocr.scene_threshold", DEFAULT_SCENE_THRESHOLD)
        ),
        "blank_std_threshold": float(
            config.get("ocr.blank_std_threshold", 0.01)
        ),
        "max_keyframes": int(config.get("ocr.max_keyframes", 0) or 0),
        "max_gap_s": float(config.get("ocr.max_gap_s", DEFAULT_MAX_GAP_S)),
        "dedupe_similarity": float(
            config.get("ocr.dedupe_similarity", DEFAULT_DEDUPE_SIMILARITY)
        ),
        "embedding_model": embedding_model,
    }


_EXTRACTION_KEYS = (
    "lang",
    "min_confidence",
    "enable_mkldnn",
    "allow_non_ascii",
    "text_detection_model_name",
    "text_recognition_model_name",
    "text_det_limit_side_len",
    "text_det_limit_type",
    "text_recognition_batch_size",
    "keyframe_min_interval_s",
    "keyframe_max_interval_s",
    "scene_threshold",
    "blank_std_threshold",
    "max_keyframes",
)


def _extraction_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: settings.get(key) for key in _EXTRACTION_KEYS}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ocr_cache_valid(
    payload: Optional[dict[str, Any]],
    metadata: dict[str, Any],
) -> bool:
    """Reuse cached keyframes whenever the source video is unchanged.

    OCR is the expensive stage, so changing keyframe/model settings only takes
    effect with ``--overwrite``. Segment policy changes still rebuild the
    index from the cached keyframes without re-running PaddleOCR.
    """
    if not payload:
        return False
    if payload.get("source_hash") != metadata.get("source_hash"):
        return False
    return "keyframes" in payload


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
    index_path, ids_path = ocr_index_paths(config, video_id)
    index_path.unlink(missing_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "layer": "ocr",
        "engine": "paddleocr",
        "lang": settings["lang"],
        "embedding_model": settings["embedding_model"],
        "device": None,
        "dim": None,
        "normalized": True,
        "count": 0,
        "empty": True,
        "reason": reason,
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "ocr_path": store.relative_path(ocr_path(config, video_id), config.repo_root),
        "settings": settings,
        "stats": {"keyframes": 0, "segments": 0, "lines": 0},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": [],
    }
    store.write_json(ids_path, payload)
    _LOG.warning("%s has no OCR text (%s); wrote an empty OCR sidecar", video_id, reason)
    return payload


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def index_ocr_video(
    video_id: str,
    config: Config,
    *,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Build (or reuse) the Layer E index for one video."""
    metadata = _read_metadata(config, video_id)
    embedding_model = str(
        config.get("text.embedding_model") or DEFAULT_TEXT_MODEL
    )
    batch_size = int(batch_size or config.get("text.batch_size", 32))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    settings = ocr_settings(config, embedding_model)
    index_path, ids_path = ocr_index_paths(config, video_id)

    if not overwrite:
        cached = _cached_index(ids_path, index_path, metadata, settings)
        if cached is not None:
            _LOG.info(
                "cache hit: %s OCR index (%d segments, %s)",
                video_id,
                int(cached.get("count", 0)),
                embedding_model,
            )
            return cached, True
        if index_path.exists() or ids_path.exists():
            _LOG.info("OCR index stale; rebuilding %s", video_id)

    ocr_file = ocr_path(config, video_id)
    extraction = _load_json(ocr_file)
    extraction_cached = not overwrite and _ocr_cache_valid(extraction, metadata)
    if extraction_cached and extraction is not None:
        stored = extraction.get("settings") or {}
        expected = _extraction_settings(settings)
        if any(
            key in stored and stored[key] != value
            for key, value in expected.items()
        ):
            _LOG.info(
                "reusing cached OCR keyframes with different settings; "
                "pass --overwrite to re-run OCR"
            )

    if not extraction_cached:
        keyframes = select_keyframes(
            metadata["frames"], config, show_progress=show_progress
        )
        if not keyframes:
            extraction = {
                "schema_version": SCHEMA_VERSION,
                "tool": "videoseek",
                "video_id": video_id,
                "source_path": metadata.get("source_path"),
                "source_hash": metadata.get("source_hash"),
                "engine": "paddleocr",
                "settings": _extraction_settings(settings),
                "stats": {"keyframes": 0, "segments": 0, "lines": 0},
                "keyframes": [],
                "segments": [],
                "created_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            store.write_json(ocr_file, extraction)
            return (
                _write_empty_sidecar(
                    video_id, config, metadata, settings, reason="no_keyframes"
                ),
                False,
            )

        engine = load_ocr_engine(config)
        min_confidence = float(settings["min_confidence"])
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                total=len(keyframes), unit="frame", desc="ocr", dynamic_ncols=True
            )
        extracted: list[dict[str, Any]] = []
        try:
            for keyframe in keyframes:
                lines = read_frame_lines(
                    engine,
                    config.resolve(keyframe["path"]),
                    min_confidence=min_confidence,
                    allow_non_ascii=bool(settings["allow_non_ascii"]),
                )
                extracted.append(
                    {
                        "timestamp_s": float(keyframe["timestamp_s"]),
                        "path": keyframe["path"],
                        "scene_diff": keyframe.get("scene_diff"),
                        "lines": lines,
                        "text": " ".join(line["text"] for line in lines).strip(),
                    }
                )
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        segments = dedupe_ocr_keyframes(
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
            "engine": "paddleocr",
            "settings": _extraction_settings(settings),
            "stats": {
                "keyframes": len(extracted),
                "keyframes_with_text": sum(1 for item in extracted if item["text"]),
                "segments": len(segments),
                "lines": sum(len(item["lines"]) for item in extracted),
            },
            "keyframes": extracted,
            "segments": segments,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        store.write_json(ocr_file, extraction)
    else:
        assert extraction is not None
        _LOG.info("cache hit: OCR extraction %s", ocr_file)

    # Rebuilding segments from cached keyframes is cheap and lets dedupe
    # settings change without re-running PaddleOCR.
    duration = float(metadata.get("duration_s") or 0.0) or None
    segments = dedupe_ocr_keyframes(
        (extraction or {}).get("keyframes") or [],
        similarity_threshold=float(settings["dedupe_similarity"]),
        max_gap_s=float(settings["max_gap_s"]),
        duration_s=duration,
    )
    if not segments:
        return (
            _write_empty_sidecar(
                video_id, config, metadata, settings, reason="no_text_detected"
            ),
            False,
        )

    text_device = resolve_device(config.get("text.device", "auto"))
    encoder = load_text_encoder(embedding_model, text_device)
    with StageTimer("embed OCR segments", _LOG):
        vectors = encode_texts(
            [segment["text"] for segment in segments],
            encoder,
            batch_size=batch_size,
        )
    if vectors.shape[0] != len(segments):
        raise RuntimeError(
            f"OCR embedding count {vectors.shape[0]} does not match segments {len(segments)}"
        )

    with StageTimer("faiss index", _LOG):
        index = store.build_flat_ip(vectors)
        store.save_faiss(index, index_path)

    records = []
    for position, segment in enumerate(segments):
        records.append(
            {
                "doc_id": f"{video_id}:ocr:{int(segment['index']):04d}",
                "video_id": video_id,
                "layer": "ocr",
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "path": segment["path"],
                "text": segment["text"],
                "meta": {
                    "segment_idx": int(segment["index"]),
                    "confidence": float(segment.get("confidence", 0.0)),
                    "line_count": len(segment.get("lines") or []),
                    "lines": segment.get("lines") or [],
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
        "layer": "ocr",
        "engine": "paddleocr",
        "embedding_model": embedding_model,
        "device": text_device,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "count": len(records),
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "ocr_path": store.relative_path(ocr_file, config.repo_root),
        "lang": settings["lang"],
        "settings": settings,
        "stats": {
            "keyframes": stats.get("keyframes", 0),
            "segments": len(segments),
            "lines": stats.get("lines", 0),
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": records,
    }
    store.write_json(ids_path, payload)
    _LOG.info(
        "wrote %d OCR segments (%d-d) from %d keyframes to %s",
        len(records),
        payload["dim"],
        stats.get("keyframes", 0),
        store.relative_path(index_path, config.repo_root),
    )
    return payload, False
