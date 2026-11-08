"""Querying for Layer A (frames) and Layer B (objects)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from src.config import Config
from src.index import store
from src.index.frames import (
    DEFAULT_MODEL,
    embed_texts,
    frame_index_paths,
    load_siglip,
    resolve_device,
)
from src.index.objects import object_index_paths
from src.utils.logging import get_logger

_LOG = get_logger("search.query")

LAYERS = ("frame", "object")

_COLOR_ALIASES = {
    "red": "red",
    "white": "white",
    "black": "black",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "grey": "grey",
    "gray": "grey",
    "silver": "grey",
}

_CLASS_KEYWORDS = {
    "van": {"car", "truck"},
    "truck": {"truck"},
    "car": {"car"},
    "suv": {"car", "truck"},
    "taxi": {"car"},
    "bus": {"bus"},
    "person": {"person"},
    "man": {"person"},
    "woman": {"person"},
    "people": {"person"},
    "cyclist": {"bicycle", "motorcycle", "person"},
    "bicycle": {"bicycle"},
    "motorcycle": {"motorcycle"},
    "backpack": {"backpack"},
    "handbag": {"handbag"},
    "suitcase": {"suitcase"},
    "bag": {"backpack", "handbag", "suitcase"},
}


def parse_object_query(text: str) -> tuple[set[str], Optional[set[str]], bool]:
    """Extract color, class, and night constraints from a natural-language query."""
    words = re.findall(r"[a-z]+", text.lower())
    colors = {_COLOR_ALIASES[word] for word in words if word in _COLOR_ALIASES}
    classes: set[str] = set()
    for word in words:
        if word in _CLASS_KEYWORDS:
            classes.update(_CLASS_KEYWORDS[word])
    night = any(word in {"night", "nighttime", "dark"} for word in words)
    return colors, (classes or None), night


def filter_object_matches(
    records: list[dict[str, Any]],
    scores: Any,
    positions: Any,
    query: str,
) -> list[tuple[float, dict[str, Any]]]:
    """Apply color/class constraints, falling back when a filter is too strict."""
    colors, classes, night = parse_object_query(query)
    pairs: list[tuple[float, dict[str, Any]]] = []
    for score, position in zip(scores, positions):
        if position < 0 or position >= len(records):
            continue
        pairs.append((float(score), records[int(position)]))

    if night:
        subset = [
            pair
            for pair in pairs
            if (pair[1].get("meta") or {}).get("brightness") is not None
            and float(pair[1]["meta"]["brightness"]) < 70
        ]
        if subset:
            pairs = subset

    if classes:
        subset = [
            pair
            for pair in pairs
            if (pair[1].get("meta") or {}).get("class_name") in classes
        ]
        if subset:
            pairs = subset

    if colors:
        subset = [
            pair
            for pair in pairs
            if (pair[1].get("meta") or {}).get("color") in colors
        ]
        if subset:
            pairs = subset

    return pairs


def list_indexed_videos(config: Config, layer: str = "frame") -> list[str]:
    """Return video ids that have a complete index for the requested layer."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; use {LAYERS}")
    paths_for = frame_index_paths if layer == "frame" else object_index_paths
    base = config.indexes_dir
    if not base.is_dir():
        return []
    videos = []
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():
            continue
        index_path, ids_path = paths_for(config, candidate.name)
        if index_path.is_file() and ids_path.is_file():
            videos.append(candidate.name)
    return videos


def _load_index(
    config: Config,
    video_id: str,
    layer: str,
) -> tuple[Any, dict[str, Any], str, str]:
    """Load an index and validate the FAISS/sidecar pair."""
    if layer == "frame":
        index_path, ids_path = frame_index_paths(config, video_id)
        hint = "scripts/index_frames.py"
        model_key = "model"
    elif layer == "object":
        index_path, ids_path = object_index_paths(config, video_id)
        hint = "scripts/index_objects.py"
        model_key = "embedding_model"
    else:
        raise ValueError(f"unknown layer {layer!r}")

    if not index_path.is_file() or not ids_path.is_file():
        if layer == "object" and ids_path.is_file():
            payload = store.read_json(ids_path)
            if payload.get("empty"):
                return None, payload, model_key, hint
        raise store.StoreError(
            f"no {layer} index for {video_id!r}; run {hint} --video <video> first"
        )
    index = store.load_faiss(index_path)
    payload = store.read_json(ids_path)
    if int(payload.get("count", -1)) != int(index.ntotal):
        raise store.StoreError(
            f"{layer} index/sidecar mismatch for {video_id!r}: "
            f"faiss={index.ntotal} sidecar={payload.get('count')}"
        )
    return index, payload, model_key, hint


def load_frame_index(config: Config, video_id: str) -> tuple[Any, dict[str, Any]]:
    index, payload, _, _ = _load_index(config, video_id, "frame")
    return index, payload


def load_object_index(config: Config, video_id: str) -> tuple[Any, dict[str, Any]]:
    index, payload, _, _ = _load_index(config, video_id, "object")
    return index, payload


def _embed_query(
    text: str,
    config: Config,
    device: Optional[str],
    model_name: Optional[str],
) -> tuple[str, str, Any]:
    resolved_model = model_name or str(config.get("frames.model") or DEFAULT_MODEL)
    resolved_device = resolve_device(device or config.get("frames.device", "auto"))
    processor, model = load_siglip(resolved_model, resolved_device)
    vector = embed_texts([text], processor, model, device=resolved_device)[0]
    return resolved_model, resolved_device, vector


def search_frames(
    query: str,
    config: Config,
    *,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Embed one query and return the top-K frame matches across videos."""
    text = query.strip()
    if not text:
        raise ValueError("query must not be empty")

    resolved_top_k = int(top_k or config.get("search.top_k", 20))
    if resolved_top_k <= 0:
        raise ValueError(f"top_k must be positive, got {resolved_top_k}")

    video_ids = [video_id] if video_id else list_indexed_videos(config, "frame")
    if not video_ids:
        raise FileNotFoundError(
            "no frame indexes found under "
            f"{config.indexes_dir}; build one with scripts/index_frames.py"
        )

    model_name, resolved_device, query_vector = _embed_query(
        text, config, device, model_name
    )

    matches: list[dict[str, Any]] = []
    total_records = 0
    for current_id in video_ids:
        index, payload = load_frame_index(config, current_id)
        total_records += int(index.ntotal)
        if payload.get("model") != model_name:
            _LOG.warning(
                "index %s was built with %s but the query uses %s; "
                "scores may be meaningless",
                current_id,
                payload.get("model"),
                model_name,
            )
        if payload.get("dim") and int(payload["dim"]) != int(index.d):
            raise store.StoreError(
                f"index {current_id!r} dimension {index.d} does not match "
                f"sidecar dimension {payload['dim']}"
            )

        scores, positions = store.search(index, query_vector, resolved_top_k)
        records = payload.get("records") or []
        for score, position in zip(scores, positions):
            if position < 0 or position >= len(records):
                continue
            record = records[int(position)]
            matches.append(
                {
                    "doc_id": record["doc_id"],
                    "video_id": current_id,
                    "layer": "frame",
                    "start": float(record["start"]),
                    "end": float(record["end"]),
                    "timestamp": float(record["start"]),
                    "path": record["path"],
                    "score": float(score),
                }
            )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": text,
        "layer": "frame",
        "model": model_name,
        "device": resolved_device,
        "videos": video_ids,
        "total_records": total_records,
        "results": matches[:resolved_top_k],
    }


def search_objects(
    query: str,
    config: Config,
    *,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Search object crops and de-duplicate multiple crops from one track."""
    text = query.strip()
    if not text:
        raise ValueError("query must not be empty")

    resolved_top_k = int(top_k or config.get("search.top_k", 20))
    if resolved_top_k <= 0:
        raise ValueError(f"top_k must be positive, got {resolved_top_k}")

    video_ids = [video_id] if video_id else list_indexed_videos(config, "object")
    if not video_ids:
        raise FileNotFoundError(
            "no object indexes found under "
            f"{config.indexes_dir}; build one with scripts/index_objects.py"
        )

    model_name, resolved_device, query_vector = _embed_query(
        text, config, device, model_name
    )

    matches: list[dict[str, Any]] = []
    total_records = 0
    for current_id in video_ids:
        index, payload = load_object_index(config, current_id)
        if index is None or int(index.ntotal) == 0:
            continue
        total_records += int(index.ntotal)
        if payload.get("embedding_model") != model_name:
            _LOG.warning(
                "object index %s was built with %s but the query uses %s",
                current_id,
                payload.get("embedding_model"),
                model_name,
            )
        if payload.get("dim") and int(payload["dim"]) != int(index.d):
            raise store.StoreError(
                f"object index {current_id!r} dimension {index.d} does not match "
                f"sidecar dimension {payload['dim']}"
            )

        fetch_k = int(index.ntotal)
        scores, positions = store.search(index, query_vector, fetch_k)
        records = payload.get("records") or []
        best_by_track: dict[int, dict[str, Any]] = {}
        for score, record in filter_object_matches(records, scores, positions, text):
            meta = record.get("meta") or {}
            track_id = int(meta.get("track_id", 0))
            candidate = {
                "doc_id": record["doc_id"],
                "video_id": current_id,
                "layer": "object",
                "start": float(record["start"]),
                "end": float(record["end"]),
                "timestamp": float(meta.get("timestamp_s", record["start"])),
                "path": record["path"],
                "score": float(score),
                "track_id": track_id,
                "class_name": meta.get("class_name"),
                "color": meta.get("color"),
                "confidence": float(meta.get("confidence", 0.0)),
            }
            existing = best_by_track.get(track_id)
            if existing is None or candidate["score"] > existing["score"]:
                best_by_track[track_id] = candidate
        matches.extend(best_by_track.values())

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": text,
        "layer": "object",
        "model": model_name,
        "device": resolved_device,
        "videos": video_ids,
        "total_records": total_records,
        "results": matches[:resolved_top_k],
    }


def find_video_id(value: str) -> str:
    """Accept either a video id or a path and return the id."""
    candidate = Path(value).expanduser()
    if candidate.is_file() or candidate.suffix:
        from src.index.video import video_id_for

        return video_id_for(candidate)
    return value
