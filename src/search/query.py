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
from src.index.ocr import ocr_index_paths
from src.index.speech import speech_index_paths
from src.index.text import (
    DEFAULT_TEXT_MODEL,
    bm25_rank,
    encode_texts,
    load_text_encoder,
    reciprocal_rank_scores,
)
from src.utils.logging import get_logger

_LOG = get_logger("search.query")

LAYERS = ("frame", "object", "speech", "ocr")

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


def _bbox_iou(first: Any, second: Any) -> float:
    if not first or not second or len(first) < 4 or len(second) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in first[:4])
    bx1, by1, bx2, by2 = (float(value) for value in second[:4])
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def collapse_object_pairs(
    pairs: list[tuple[float, dict[str, Any]]],
    *,
    window_s: float = 2.0,
    iou_threshold: float = 0.3,
) -> list[tuple[float, dict[str, Any]]]:
    """Collapse repeated anchor crops of the same physical object."""
    ordered = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    kept: list[tuple[float, dict[str, Any]]] = []
    for score, record in ordered:
        meta = record.get("meta") or {}
        timestamp = float(meta.get("timestamp_s", 0.0))
        box = meta.get("crop_bbox") or meta.get("bbox")
        duplicate = False
        for _, other in kept:
            other_meta = other.get("meta") or {}
            if other_meta.get("class_name") != meta.get("class_name"):
                continue
            if abs(float(other_meta.get("timestamp_s", 0.0)) - timestamp) > window_s:
                continue
            other_box = other_meta.get("crop_bbox") or other_meta.get("bbox")
            if _bbox_iou(box, other_box) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, record))
    return kept


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

    return collapse_object_pairs(pairs)


def list_indexed_videos(config: Config, layer: str = "frame") -> list[str]:
    """Return video ids that have a complete index for the requested layer."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; use {LAYERS}")
    paths_for = {
        "frame": frame_index_paths,
        "object": object_index_paths,
        "speech": speech_index_paths,
        "ocr": ocr_index_paths,
    }[layer]
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
    elif layer == "speech":
        index_path, ids_path = speech_index_paths(config, video_id)
        hint = "scripts/index_speech.py"
        model_key = "embedding_model"
    elif layer == "ocr":
        index_path, ids_path = ocr_index_paths(config, video_id)
        hint = "scripts/index_ocr.py"
        model_key = "embedding_model"
    else:
        raise ValueError(f"unknown layer {layer!r}")

    if not index_path.is_file() or not ids_path.is_file():
        if layer in ("object", "speech", "ocr") and ids_path.is_file():
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


def load_speech_index(config: Config, video_id: str) -> tuple[Any, dict[str, Any]]:
    index, payload, _, _ = _load_index(config, video_id, "speech")
    return index, payload


def load_ocr_index(config: Config, video_id: str) -> tuple[Any, dict[str, Any]]:
    index, payload, _, _ = _load_index(config, video_id, "ocr")
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


def rank_text_hits(
    records: list[dict[str, Any]],
    scores: Any,
    positions: Any,
    query_text: str,
    *,
    layer: str,
    video_id: str,
    use_bm25: bool = True,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense and BM25 rankings for a text layer into RRF hits."""
    texts = [str(record.get("text") or "") for record in records]

    dense_pairs: list[tuple[int, float]] = []
    for score, position in zip(scores, positions):
        index = int(position)
        if index < 0 or index >= len(records):
            continue
        dense_pairs.append((index, float(score)))
    dense_pairs.sort(key=lambda pair: pair[1], reverse=True)
    dense_ranked = [index for index, _ in dense_pairs]
    dense_scores = {index: score for index, score in dense_pairs}

    sparse_ranked: list[int] = []
    sparse_scores: dict[int, float] = {}
    if use_bm25 and any(texts):
        sparse_pairs = bm25_rank(query_text, texts)
        sparse_ranked = [index for index, _ in sparse_pairs]
        sparse_scores = {index: score for index, score in sparse_pairs}

    rankings = [ranking for ranking in (dense_ranked, sparse_ranked) if ranking]
    combined = reciprocal_rank_scores(rankings, k=rrf_k)

    hits: list[dict[str, Any]] = []
    for position, score in sorted(combined.items(), key=lambda pair: pair[1], reverse=True):
        record = records[position]
        hits.append(
            {
                "doc_id": record["doc_id"],
                "video_id": video_id,
                "layer": layer,
                "start": float(record["start"]),
                "end": float(record["end"]),
                "segment_end": float(record["end"]),
                "timestamp": float(record["start"]),
                "path": record.get("path"),
                "text": record.get("text"),
                "score": float(score),
                "dense_score": dense_scores.get(position),
                "bm25_score": sparse_scores.get(position),
            }
        )
    return hits


def rank_speech_hits(
    records: list[dict[str, Any]],
    scores: Any,
    positions: Any,
    query_text: str,
    *,
    video_id: str,
    use_bm25: bool = True,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    return rank_text_hits(
        records,
        scores,
        positions,
        query_text,
        layer="speech",
        video_id=video_id,
        use_bm25=use_bm25,
        rrf_k=rrf_k,
    )


def rank_ocr_hits(
    records: list[dict[str, Any]],
    scores: Any,
    positions: Any,
    query_text: str,
    *,
    video_id: str,
    use_bm25: bool = True,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    return rank_text_hits(
        records,
        scores,
        positions,
        query_text,
        layer="ocr",
        video_id=video_id,
        use_bm25=use_bm25,
        rrf_k=rrf_k,
    )


def _search_text_layer(
    query: str,
    config: Config,
    *,
    layer: str,
    hint: str,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    use_bm25: Optional[bool] = None,
) -> dict[str, Any]:
    text = query.strip()
    if not text:
        raise ValueError("query must not be empty")

    resolved_top_k = int(top_k or config.get("search.top_k", 20))
    if resolved_top_k <= 0:
        raise ValueError(f"top_k must be positive, got {resolved_top_k}")

    video_ids = [video_id] if video_id else list_indexed_videos(config, layer)
    if not video_ids:
        raise FileNotFoundError(
            f"no {layer} indexes found under {config.indexes_dir}; "
            f"build one with {hint}"
        )

    resolved_model = str(
        model_name or config.get("text.embedding_model") or DEFAULT_TEXT_MODEL
    )
    resolved_device = resolve_device(device or config.get("text.device", "auto"))
    encoder = load_text_encoder(resolved_model, resolved_device)
    query_vector = encode_texts(
        [text],
        encoder,
        batch_size=int(config.get("text.batch_size", 32)),
        query=True,
        query_prefix=bool(config.get("text.query_prefix", True)),
    )[0]

    if use_bm25 is None:
        use_bm25 = bool(config.get("text.use_bm25", True))
    rrf_k = int(config.get("search.rrf_k", 60))

    matches: list[dict[str, Any]] = []
    total_records = 0
    for current_id in video_ids:
        index, payload, _, _ = _load_index(config, current_id, layer)
        if index is None or int(index.ntotal) == 0:
            continue
        total_records += int(index.ntotal)
        if payload.get("embedding_model") != resolved_model:
            _LOG.warning(
                "%s index %s was built with %s but the query uses %s",
                layer,
                current_id,
                payload.get("embedding_model"),
                resolved_model,
            )
        if payload.get("dim") and int(payload["dim"]) != int(index.d):
            raise store.StoreError(
                f"{layer} index {current_id!r} dimension {index.d} does not match "
                f"sidecar dimension {payload['dim']}"
            )

        scores, positions = store.search(index, query_vector, int(index.ntotal))
        matches.extend(
            rank_text_hits(
                payload.get("records") or [],
                scores,
                positions,
                text,
                layer=layer,
                video_id=current_id,
                use_bm25=use_bm25,
                rrf_k=rrf_k,
            )
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": text,
        "layer": layer,
        "model": resolved_model,
        "device": resolved_device,
        "videos": video_ids,
        "total_records": total_records,
        "results": matches[:resolved_top_k],
    }


def search_speech(
    query: str,
    config: Config,
    *,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    use_bm25: Optional[bool] = None,
) -> dict[str, Any]:
    """Search the speech transcript layer (dense + BM25, fused with RRF)."""
    return _search_text_layer(
        query,
        config,
        layer="speech",
        hint="scripts/index_speech.py",
        video_id=video_id,
        top_k=top_k,
        device=device,
        model_name=model_name,
        use_bm25=use_bm25,
    )


def search_ocr(
    query: str,
    config: Config,
    *,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    use_bm25: Optional[bool] = None,
) -> dict[str, Any]:
    """Search the on-screen-text layer (dense + BM25, fused with RRF)."""
    return _search_text_layer(
        query,
        config,
        layer="ocr",
        hint="scripts/index_ocr.py",
        video_id=video_id,
        top_k=top_k,
        device=device,
        model_name=model_name,
        use_bm25=use_bm25,
    )


def find_video_id(value: str) -> str:
    """Accept either a video id or a path and return the id."""
    candidate = Path(value).expanduser()
    if candidate.is_file() or candidate.suffix:
        from src.index.video import video_id_for

        return video_id_for(candidate)
    return value
