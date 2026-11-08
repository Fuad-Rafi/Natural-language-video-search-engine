"""Temporal merging: turn ranked hits into timestamped segments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

DEFAULT_GAP_S = 3.0
DEFAULT_MIN_SEGMENT_S = 1.0


def hit_timestamp(hit: dict[str, Any]) -> float:
    if hit.get("timestamp") is not None:
        return float(hit["timestamp"])
    if hit.get("peak_timestamp") is not None:
        return float(hit["peak_timestamp"])
    return float(hit.get("start", 0.0))


def merge_hits(
    hits: Iterable[dict[str, Any]],
    *,
    gap_s: float = DEFAULT_GAP_S,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    max_segment_s: float | None = None,
) -> list[dict[str, Any]]:
    """Group hits into segments where consecutive hits are within ``gap_s``.

    Segments are returned ordered by peak score (best first). Each segment
    carries its contributing layers so the UI can render source badges.
    """
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        if hit.get("video_id") is None:
            continue
        by_video[str(hit["video_id"])].append(hit)

    segments: list[dict[str, Any]] = []
    for video_id, group in by_video.items():
        group.sort(key=hit_timestamp)
        current: list[dict[str, Any]] = []
        for hit in group:
            if current and hit_timestamp(hit) - hit_timestamp(current[-1]) > gap_s:
                segments.extend(
                    _build_segments(video_id, current, min_segment_s, max_segment_s)
                )
                current = []
            current.append(hit)
        if current:
            segments.extend(
                _build_segments(video_id, current, min_segment_s, max_segment_s)
            )

    segments.sort(key=lambda segment: segment["score"], reverse=True)
    return segments


def _build_segments(
    video_id: str,
    hits: list[dict[str, Any]],
    min_segment_s: float,
    max_segment_s: float | None,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    if max_segment_s and max_segment_s > 0:
        start = hit_timestamp(hits[0])
        chunks: list[list[dict[str, Any]]] = [[]]
        for hit in hits:
            if chunks[-1] and hit_timestamp(hit) - start > max_segment_s:
                chunks.append([])
                start = hit_timestamp(hit)
            chunks[-1].append(hit)
        if len(chunks) > 1:
            built: list[dict[str, Any]] = []
            for chunk in chunks:
                built.extend(_build_segments(video_id, chunk, min_segment_s, None))
            return built

    peak = max(hits, key=lambda hit: float(hit.get("score", 0.0)))
    start = hit_timestamp(hits[0])
    end = max(hit_timestamp(hits[-1]), start + max(0.0, min_segment_s))
    layers = sorted(
        {
            layer
            for hit in hits
            for layer in (hit.get("layers") or [hit.get("layer")])
            if layer
        }
    )
    peak_timestamp = hit_timestamp(peak)
    return [
        {
            "video_id": video_id,
            "start": float(start),
            "end": float(end),
            "score": float(peak.get("score", 0.0)),
            "peak_timestamp": float(peak_timestamp),
            "layers": layers,
            "hit_count": len(hits),
            "thumbnail": peak.get("thumbnail") or peak.get("path"),
            "peak_path": peak.get("thumbnail") or peak.get("path"),
            "doc_id": peak.get("doc_id"),
            "evidence": peak,
        }
    ]


def duplicate_rate(
    items: Iterable[dict[str, Any]],
    *,
    gap_s: float = DEFAULT_GAP_S,
    top_k: int = 10,
) -> float:
    """Fraction of top-k results that repeat a nearby result in the same video."""
    top = list(items)[:top_k]
    if len(top) < 2:
        return 0.0
    duplicates = 0
    for index, item in enumerate(top):
        timestamp = hit_timestamp(item)
        video_id = str(item.get("video_id"))
        if any(
            str(previous.get("video_id")) == video_id
            and abs(hit_timestamp(previous) - timestamp) < gap_s
            for previous in top[:index]
        ):
            duplicates += 1
    return duplicates / len(top)
