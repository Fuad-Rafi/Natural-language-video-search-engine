"""High-level search service shared by the API and the Gradio UI.

The evaluator applies benchmark-specific gating. The UI needs a simpler
contract: a query, an optional video, and a layer choice. ``auto`` routes by
query cues, a specific layer searches only that layer, and ``all`` fuses every
available layer with RRF before merging into timestamped segments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from src.config import Config
from src.index import store
from src.search.fusion import reciprocal_rank_fusion
from src.search.query import (
    list_indexed_videos,
    parse_object_query,
    search_caption,
    search_frames,
    search_objects,
    search_ocr,
    search_speech,
)
from src.search.temporal import merge_hits
from src.utils.logging import get_logger

_LOG = get_logger("search.service")

LAYER_CHOICES = ("auto", "frames", "objects", "speech", "ocr", "caption", "all")
LAYER_KEY = {
    "frames": "frame",
    "objects": "object",
    "speech": "speech",
    "ocr": "ocr",
    "caption": "caption",
}
LAYER_BADGE = {
    "frame": "VISUAL",
    "object": "OBJECT",
    "speech": "SPEECH",
    "ocr": "TEXT",
    "caption": "CAPTION",
}

_SPEECH_CUES = (
    "say", "says", "said", "mention", "mentions", "talk", "talks",
    "explain", "explains", "transcript", "spoken", "voice", "where he",
    "where she", "what did", "word", "phrase",
)
_OCR_CUES = (
    "slide", "sign", "text", "written", "screen", "title", "command",
    "error", "menu", "label", "prompt", "chalkboard", "whiteboard",
)
_CAPTION_CUES = (
    "action", "doing", "argue", "arguing", "play", "playing", "walk",
    "walking", "making", "holding", "sitting", "standing", "running",
    "cooking", "driving", "family", "person", "people", "man", "woman",
)


def _has_cue(text: str, words: set[str], cues: Iterable[str]) -> bool:
    return any(
        (cue in text) if " " in cue else (cue in words)
        for cue in cues
    )


def infer_layers(query: str, available: Optional[Iterable[str]] = None) -> list[str]:
    """Choose layers from query cues. Frames are always included.

    OCR and speech are the most specific cues, so they win over object/caption
    words such as "people" or "man". Use ``layers="all"`` to override.
    """
    text = query.lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    if _has_cue(text, words, _OCR_CUES):
        chosen = {"ocr"}
    elif _has_cue(text, words, _SPEECH_CUES):
        chosen = {"speech"}
    else:
        chosen = {"frame"}
        colors, classes, night = parse_object_query(query)
        if classes or colors:
            chosen.add("object")
        if _has_cue(text, words, _CAPTION_CUES):
            chosen.add("caption")
    if available is not None:
        allowed = {LAYER_KEY.get(name, name) for name in available}
        chosen &= allowed | {"frame"}
    if not chosen:
        chosen = {"frame"}
    return sorted(chosen)


def _normalise_hits(
    payload: dict[str, Any], *, layer_key: str, video_id: Optional[str]
) -> list[dict[str, Any]]:
    hits = []
    for item in payload.get("results", []):
        hit = dict(item)
        hit["video_id"] = hit.get("video_id") or video_id
        hit["layer"] = layer_key
        if layer_key == "object":
            label = hit.get("class_name") or "object"
            if hit.get("color"):
                label = f"{label} ({hit['color']})"
            hit.setdefault("text", label)
        hits.append(hit)
    return hits


def _search_layer(
    query: str,
    config: Config,
    layer_key: str,
    *,
    video_id: Optional[str],
    candidates: int,
    device: Optional[str],
    use_bm25: Optional[bool],
) -> list[dict[str, Any]]:
    search = {
        "frame": search_frames,
        "object": search_objects,
        "speech": search_speech,
        "ocr": search_ocr,
        "caption": search_caption,
    }[layer_key]
    kwargs: dict[str, Any] = {
        "video_id": video_id,
        "top_k": candidates,
        "device": device,
    }
    if layer_key in ("speech", "ocr", "caption") and use_bm25 is not None:
        kwargs["use_bm25"] = use_bm25
    try:
        payload = search(query, config, **kwargs)
    except (FileNotFoundError, store.StoreError) as exc:
        _LOG.info("skipping %s: %s", layer_key, exc)
        return []
    return _normalise_hits(payload, layer_key=layer_key, video_id=video_id)


def _video_metadata(config: Config, video_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(video_id)):
        return {}
    path = config.frames_dir / video_id / "metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _nearest_frame_path(metadata: dict[str, Any], timestamp: float) -> Optional[str]:
    frames = metadata.get("frames") or []
    if not frames:
        return None
    nearest = min(
        frames,
        key=lambda frame: abs(float(frame.get("timestamp_s", 0.0)) - timestamp),
    )
    return nearest.get("path")


def search(
    query: str,
    config: Config,
    *,
    video_id: Optional[str] = None,
    top_k: int = 10,
    layers: str | Iterable[str] = "auto",
    candidates: int = 50,
    device: Optional[str] = None,
    use_bm25: Optional[bool] = None,
    adaptive: bool = True,
) -> dict[str, Any]:
    """Run a UI search and return ranked, enriched segments."""
    text = query.strip()
    if not text:
        raise ValueError("query must not be empty")
    top_k = max(1, int(top_k))

    if isinstance(layers, str):
        requested = layers
    else:
        requested = list(layers)
    if requested in ("auto", ["auto"]):
        layer_keys = infer_layers(text)
    elif requested in ("all", ["all"]):
        layer_keys = ["frame", "object", "speech", "ocr", "caption"]
    elif isinstance(requested, str):
        layer_keys = [LAYER_KEY.get(requested, requested)]
    else:
        layer_keys = [LAYER_KEY.get(str(item), str(item)) for item in requested]

    hits_by_layer: dict[str, list[dict[str, Any]]] = {}
    for layer_key in layer_keys:
        hits = _search_layer(
            text,
            config,
            layer_key,
            video_id=video_id,
            candidates=candidates,
            device=device,
            use_bm25=use_bm25,
        )
        if hits:
            hits_by_layer[layer_key] = hits

    if not hits_by_layer:
        return {
            "query": text,
            "layers": layer_keys,
            "videos": [],
            "segments": [],
            "count": 0,
            "adaptive": adaptive,
            "strong": False,
        }

    weights = {
        key: float(config.get(f"search.weights.{key}", 1.0))
        for key in ("frame", "object", "speech", "ocr", "caption")
    }
    if len(hits_by_layer) == 1:
        raw = next(iter(hits_by_layer.values()))
    else:
        raw = reciprocal_rank_fusion(
            hits_by_layer,
            k=int(config.get("search.rrf_k", 60)),
            weights=weights,
            window_s=float(config.get("search.rrf_window_s", 2.0)),
            max_per_layer=candidates,
        )

    segments = select_adaptive(
        merge_hits(
            raw,
            gap_s=float(config.get("search.merge_gap_s", 3.0)),
            min_segment_s=float(config.get("search.min_segment_s", 1.0)),
        ),
        config,
        max_results=top_k,
        enabled=adaptive,
    )
    strong = bool(segments) and not any(
        segment.get("weak") for segment in segments
    )

    video_ids = sorted({segment["video_id"] for segment in segments})
    metadata = {vid: _video_metadata(config, vid) for vid in video_ids}
    enriched = []
    for segment in segments:
        evidence = segment.get("evidence") or {}
        video_meta = metadata.get(segment["video_id"], {})
        peak = float(segment.get("peak_timestamp", segment.get("start", 0.0)))
        path = segment.get("peak_path") or segment.get("thumbnail")
        if not path:
            path = _nearest_frame_path(video_meta, peak)
        text_hit = evidence.get("text")
        enriched.append(
            {
                **segment,
                "badges": [
                    LAYER_BADGE.get(layer, layer.upper())
                    for layer in segment.get("layers", [])
                ],
                "thumbnail_url": f"/media/{path}" if path else None,
                "clip_url": (
                    f"/clip?video_id={segment['video_id']}"
                    f"&start={float(segment['start']):.2f}"
                    f"&end={float(segment['end']):.2f}"
                ),
                "text": text_hit,
                "source_path": video_meta.get("source_path"),
                "duration_s": video_meta.get("duration_s"),
            }
        )

    return {
        "query": text,
        "layers": sorted(hits_by_layer),
        "videos": video_ids,
        "segments": enriched,
        "count": len(enriched),
        "adaptive": adaptive,
        "strong": strong,
    }


def list_videos(config: Config) -> list[dict[str, Any]]:
    """Return indexed videos with their available layers."""
    by_video: dict[str, set[str]] = {}
    for layer in ("frame", "object", "speech", "ocr", "caption"):
        for video_id in list_indexed_videos(config, layer):
            by_video.setdefault(video_id, set()).add(layer)
    videos = []
    for video_id, layers in sorted(by_video.items()):
        metadata = _video_metadata(config, video_id)
        videos.append(
            {
                "video_id": video_id,
                "layers": sorted(layers),
                "duration_s": metadata.get("duration_s"),
                "source_path": metadata.get("source_path"),
                "has_audio": metadata.get("has_audio"),
            }
        )
    return videos


def is_selectable_source(source: Optional[str], config: Config) -> bool:
    """True when a video's source file exists and is not benchmark data.

    Dead indexes (source deleted) cannot play clips, and the public Charades-STA
    subset under ``data/public`` is for offline evaluation rather than the
    interactive picker.
    """
    if not source:
        return False
    path = config.resolve(source).resolve()
    if not path.is_file():
        return False
    public_root = (config.data_dir / "public").resolve()
    return not path.is_relative_to(public_root)


def selectable_videos(
    config: Config, videos: Optional[Iterable[dict[str, Any]]] = None
) -> list[dict[str, Any]]:
    """Return indexed videos suitable for the interactive video picker."""
    items = list_videos(config) if videos is None else list(videos)
    return [
        video
        for video in items
        if is_selectable_source(video.get("source_path"), config)
    ]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 1.0 if value >= ceiling else 0.0
    return _clamp01((value - floor) / (ceiling - floor))


def _text_relevance(
    dense: Optional[float], bm25: Optional[float], config: Config
) -> float:
    """Blend the dense cosine with the lexical BM25 signal."""
    bm25_ceiling = float(config.get("search.relevance.bm25_ceiling", 12.0))
    if dense is None:
        return _normalise(float(bm25 or 0.0), 0.0, bm25_ceiling)
    dense_rel = _normalise(
        float(dense),
        float(config.get("search.relevance.dense_floor", 0.42)),
        float(config.get("search.relevance.dense_ceiling", 0.75)),
    )
    lexical_rel = _normalise(float(bm25 or 0.0), 0.0, bm25_ceiling)
    weight = float(config.get("search.relevance.dense_weight", 0.8))
    return _clamp01(weight * dense_rel + (1.0 - weight) * lexical_rel)


def segment_relevance(segment: dict[str, Any], config: Config) -> float:
    """Estimate how strongly a segment matches the query, on a 0-1 scale.

    Layers use different score scales, so relevance is derived from the
    evidence the layer actually produced: SigLIP cosine for frames/objects,
    dense cosine plus BM25 for text layers, and the best of both for a fused
    cluster.
    """
    evidence = segment.get("evidence") or {}
    cosine = evidence.get("best_cosine")
    dense = evidence.get("best_dense")
    bm25 = evidence.get("best_bm25")
    if cosine is not None or dense is not None or bm25 is not None:
        visual = (
            _normalise(
                float(cosine),
                float(config.get("search.relevance.cosine_floor", 0.075)),
                float(config.get("search.relevance.cosine_ceiling", 0.13)),
            )
            if cosine is not None
            else 0.0
        )
        textual = (
            _text_relevance(dense, bm25, config)
            if dense is not None or bm25 is not None
            else 0.0
        )
        return max(visual, textual)

    layer = evidence.get("layer")
    if layer in ("frame", "object"):
        return _normalise(
            float(evidence.get("score", 0.0)),
            float(config.get("search.relevance.cosine_floor", 0.075)),
            float(config.get("search.relevance.cosine_ceiling", 0.13)),
        )
    if layer in ("speech", "ocr", "caption"):
        return _text_relevance(
            evidence.get("dense_score"), evidence.get("bm25_score"), config
        )
    return 0.0


def select_adaptive(
    segments: list[dict[str, Any]],
    config: Config,
    *,
    max_results: int,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Keep only relevant segments, never padding to ``max_results``.

    Always returns at least the top-ranked segment; if nothing clears the
    relevance floor it is marked ``weak`` so the UI can say "best guess".
    """
    if not segments:
        return []
    for segment in segments:
        segment["relevance"] = round(segment_relevance(segment, config), 4)

    limit = max(1, int(max_results))
    if not enabled:
        return segments[:limit]

    floor = float(config.get("search.relevance.min", 0.30))
    ratio = float(config.get("search.relevance.ratio", 0.55))
    best = max(float(segment["relevance"]) for segment in segments)
    kept: list[dict[str, Any]] = []
    for segment in segments:
        if len(kept) >= limit:
            break
        relevance = float(segment["relevance"])
        if relevance >= floor and relevance >= ratio * best:
            kept.append(segment)

    if not kept:
        guess = segments[0].copy()
        guess["weak"] = True
        kept = [guess]

    for segment in kept:
        segment.setdefault("weak", False)
    return kept


def resolve_source(config: Config, video_id: str) -> Optional[Path]:
    metadata = _video_metadata(config, video_id)
    source = metadata.get("source_path")
    if not source:
        return None
    path = config.resolve(source)
    return path if path.is_file() else None
