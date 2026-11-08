"""Reciprocal Rank Fusion across retrieval layers.

Scores from different encoders are not comparable, so fusion uses ranks:
``score(d) = Σ weight_layer / (k + rank_layer(d))``. Hits from different
layers at the same moment are clustered before scoring, so a frame hit and an
object crop at 190 s reinforce each other instead of competing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

DEFAULT_K = 60
DEFAULT_WINDOW_S = 2.0
DEFAULT_MAX_PER_LAYER = 50


def hit_timestamp(hit: dict[str, Any]) -> float:
    if hit.get("timestamp") is not None:
        return float(hit["timestamp"])
    return float(hit.get("start", 0.0))


def reciprocal_rank_fusion(
    rankings: dict[str, Iterable[dict[str, Any]]],
    *,
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    max_per_layer: int = DEFAULT_MAX_PER_LAYER,
) -> list[dict[str, Any]]:
    """Fuse per-layer ranked lists into moment clusters sorted by RRF score."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    weights = weights or {}

    points: list[dict[str, Any]] = []
    for layer, hits in rankings.items():
        for rank, hit in enumerate(list(hits)[:max_per_layer], start=1):
            if hit.get("video_id") is None:
                continue
            points.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "weight": float(weights.get(layer, 1.0)),
                    "hit": hit,
                    "timestamp": hit_timestamp(hit),
                    "video_id": str(hit["video_id"]),
                }
            )

    if not points:
        return []

    points.sort(key=lambda point: (point["video_id"], point["timestamp"]))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for point in points:
        if (
            current
            and point["video_id"] == current[-1]["video_id"]
            and point["timestamp"] - current[-1]["timestamp"] <= window_s
        ):
            current.append(point)
        else:
            if current:
                clusters.append(current)
            current = [point]
    if current:
        clusters.append(current)

    fused: list[dict[str, Any]] = []
    for cluster in clusters:
        best_rank: dict[str, int] = {}
        for point in cluster:
            layer = point["layer"]
            best_rank[layer] = min(best_rank.get(layer, 10**9), point["rank"])
        score = sum(
            float(weights.get(layer, 1.0)) / (k + rank)
            for layer, rank in best_rank.items()
        )
        peak_point = max(cluster, key=lambda point: float(point["hit"].get("score", 0.0)))
        peak = peak_point["hit"]
        timestamps = [point["timestamp"] for point in cluster]
        fused.append(
            {
                "video_id": cluster[0]["video_id"],
                "timestamp": hit_timestamp(peak),
                "start": min(timestamps),
                "end": max(timestamps),
                "score": float(score),
                "layer": "fusion",
                "layers": sorted(best_rank),
                "thumbnail": peak.get("path") or peak.get("thumbnail"),
                "path": peak.get("path") or peak.get("thumbnail"),
                "doc_id": peak.get("doc_id"),
                "evidence": peak,
                "hit_count": len(cluster),
            }
        )

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def layer_rankings(
    frame_hits: Iterable[dict[str, Any]],
    object_hits: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {"frame": list(frame_hits), "object": list(object_hits)}
