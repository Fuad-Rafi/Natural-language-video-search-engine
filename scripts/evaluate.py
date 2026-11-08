#!/usr/bin/env python
"""P2 evaluation harness for the frozen custom query set.

Computes Recall@1/5/10, mIoU, and query latency for Layer A (frames).
Layer A returns a single frame, so mIoU uses a 1-second predicted window
centered on the top hit (``search.min_segment_s`` = 1.0). That is the honest
baseline before temporal merging (P4).

Example:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index import store  # noqa: E402
from src.index.frames import embed_texts, load_siglip, resolve_device  # noqa: E402
from src.search.query import (  # noqa: E402
    filter_object_matches,
    load_frame_index,
    load_object_index,
)
from src.utils.logging import format_duration, get_logger, setup_logging  # noqa: E402

METRIC_KS = (1, 5, 10)
DEFAULT_TOLERANCE_S = 0.5
DEFAULT_MIN_SEGMENT_S = 1.0

_CAUSE_BY_TYPE = {
    "speech": "speech layer not implemented (P5)",
    "ocr": "OCR layer not implemented (P6)",
    "object": "small object missed by frame embedding (P3)",
    "spatial": "no spatial reasoning (P3)",
    "action": "frame embedding cannot model actions (P7)",
    "caption": "caption layer not implemented (P7)",
    "reasoning": "caption layer not implemented (P7)",
    "visual": "scene confusion in frame embedding",
    "counting": "frame embedding cannot count reliably",
}

_LOG = get_logger("evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Layer A on the frozen query set.")
    parser.add_argument("--queries", default="benchmarks/queries.csv")
    parser.add_argument(
        "--layer",
        default="frames",
        choices=["frames", "objects"],
        help="retrieval layer to evaluate (default: frames)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--failures-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def read_queries(path: Path) -> list[dict[str, Any]]:
    """Read and validate the frozen query CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"query file not found: {path}")
    required = {"query_id", "video_id", "query", "type", "start", "end"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"query CSV is missing columns: {', '.join(sorted(missing))}")
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            try:
                start = float(raw["start"])
                end = float(raw["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: start/end must be numbers") from exc
            if end < start:
                raise ValueError(f"line {line_number}: end < start")
            row = {
                "query_id": raw["query_id"].strip(),
                "video_id": raw["video_id"].strip(),
                "query": raw["query"].strip(),
                "type": (raw.get("type") or "visual").strip().lower(),
                "start": start,
                "end": end,
                "notes": (raw.get("notes") or "").strip(),
                "line": line_number,
            }
            if not row["query_id"] or not row["video_id"] or not row["query"]:
                raise ValueError(f"line {line_number}: empty id/video/query")
            rows.append(row)
    if not rows:
        raise ValueError(f"no queries found in {path}")
    return rows


def group_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group repeated query ids so multiple GT intervals evaluate the best one."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["query_id"], row["video_id"])
        if key not in groups:
            groups[key] = {
                "query_id": row["query_id"],
                "video_id": row["video_id"],
                "query": row["query"],
                "type": row["type"],
                "notes": row["notes"],
                "intervals": [],
            }
        groups[key]["intervals"].append((row["start"], row["end"]))
    return list(groups.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predicted_window(timestamp: float, min_segment_s: float) -> tuple[float, float]:
    half = max(min_segment_s, 0.0) / 2.0
    return timestamp - half, timestamp + half


def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def is_hit(timestamp: float, intervals: list[tuple[float, float]], tolerance: float) -> bool:
    return any(start - tolerance <= timestamp <= end + tolerance for start, end in intervals)


def best_iou(timestamp: float, intervals: list[tuple[float, float]], min_segment_s: float) -> float:
    pred_start, pred_end = predicted_window(timestamp, min_segment_s)
    return max(
        temporal_iou(pred_start, pred_end, start, end) for start, end in intervals
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    summary: dict[str, Any] = {"queries": count}
    for k in METRIC_KS:
        hits = sum(1 for result in results if result["hits"][k])
        summary[f"recall@{k}"] = hits / count if count else 0.0
    summary["miou"] = (
        sum(result["miou"] for result in results) / count if count else 0.0
    )
    latencies = [result["latency_ms"] for result in results]
    summary["latency_p50_ms"] = percentile(latencies, 0.50)
    summary["latency_p95_ms"] = percentile(latencies, 0.95)
    return summary


def build_gallery(
    failures: list[dict[str, Any]],
    config: Any,
    out_dir: Path,
    max_items: int,
) -> Path | None:
    """Render up to 20 failed top-1 results into one labeled contact sheet."""
    tiles = []
    for failure in failures[:max_items]:
        if not failure.get("predicted_path"):
            continue
        frame_path = config.resolve(failure["predicted_path"])
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile_width, tile_height = 320, 180
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros((tile_height + 58, tile_width, 3), dtype=np.uint8)
        offset_x = (tile_width - resized.shape[1]) // 2
        offset_y = (tile_height - resized.shape[0]) // 2
        tile[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized

        gt = failure["intervals"][0]
        lines = [
            failure["query"][:44],
            f"pred {failure['top1_timestamp']:.1f}s | GT {gt[0]:.1f}-{gt[1]:.1f}s",
            failure["cause"][:52],
        ]
        for row, text in enumerate(lines):
            cv2.putText(
                tile,
                text,
                (5, tile_height + 15 + row * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        tiles.append(tile)

    if not tiles:
        return None

    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack(
        [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = out_dir / "gallery.jpg"
    if not cv2.imwrite(str(gallery_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        return None
    return gallery_path


def write_markdown(
    path: Path,
    payload: dict[str, Any],
) -> None:
    aggregate = payload["aggregate"]
    lines = [
        f"# Baseline results — {payload['config']}",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Model: `{payload['model']}`",
        f"- Layer: `{payload['layer']}`",
        f"- Queries: {aggregate['queries']} across {len(payload['per_video'])} videos",
        f"- Query file SHA-256: `{payload['queries_sha256'][:16]}…`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@1 | {aggregate['recall@1']:.3f} |",
        f"| Recall@5 | {aggregate['recall@5']:.3f} |",
        f"| Recall@10 | {aggregate['recall@10']:.3f} |",
        f"| mIoU (1 s window) | {aggregate['miou']:.3f} |",
        f"| Latency p50 | {aggregate['latency_p50_ms']:.0f} ms |",
        f"| Latency p95 | {aggregate['latency_p95_ms']:.0f} ms |",
        "",
        "## Per query type",
        "",
        "| Type | Queries | R@1 | R@5 | R@10 | mIoU |",
        "|---|---|---|---|---|---|",
    ]
    for query_type, summary in sorted(payload["per_type"].items()):
        lines.append(
            f"| {query_type} | {summary['queries']} | {summary['recall@1']:.3f} | "
            f"{summary['recall@5']:.3f} | {summary['recall@10']:.3f} | {summary['miou']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per video",
            "",
            "| Video | Queries | R@1 | R@5 | mIoU |",
            "|---|---|---|---|---|",
        ]
    )
    for video_id, summary in sorted(payload["per_video"].items()):
        lines.append(
            f"| {video_id} | {summary['queries']} | {summary['recall@1']:.3f} | "
            f"{summary['recall@5']:.3f} | {summary['miou']:.3f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    queries_path = config.resolve(args.queries)
    layer = args.layer
    if layer == "frames":
        default_out = "benchmarks/results/baseline.json"
        default_markdown = "benchmarks/results/baseline.md"
        default_failures = "benchmarks/failures"
    else:
        default_out = "benchmarks/results/objects.json"
        default_markdown = "benchmarks/results/objects.md"
        default_failures = "benchmarks/failures/objects"
    out_path = config.resolve(args.out or default_out)
    markdown_path = config.resolve(args.markdown or default_markdown)
    failures_dir = config.resolve(args.failures_dir or default_failures)
    log_path = config.logs_dir / f"evaluate-{datetime.now():%Y%m%d-%H%M%S}.log"
    setup_logging(log_path, level=getattr(logging, args.log_level), console=True)

    try:
        rows = read_queries(queries_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    groups = group_queries(rows)
    if args.top_k < max(METRIC_KS):
        print(f"error: --top-k must be at least {max(METRIC_KS)}", file=sys.stderr)
        return 2

    video_ids = sorted({group["video_id"] for group in groups})
    device = resolve_device(args.device or config.get("frames.device", "auto"))
    model_name = str(config.get("frames.model"))

    _LOG.info("loading %s on %s", model_name, device)
    processor, model = load_siglip(model_name, device)

    indexes: dict[str, tuple[Any, dict[str, Any]]] = {}
    loader = load_frame_index if layer == "frames" else load_object_index
    for video_id in video_ids:
        _LOG.info("loading %s index for %s", layer, video_id)
        indexes[video_id] = loader(config, video_id)

    # Warm up CUDA so the first measured query is not inflated.
    embed_texts(["warmup"], processor, model, device=device)

    results: list[dict[str, Any]] = []
    for number, group in enumerate(groups, start=1):
        index, sidecar = indexes[group["video_id"]]
        records = sidecar["records"]

        query_started = time.perf_counter()
        vector = embed_texts([group["query"]], processor, model, device=device)[0]
        if index is None or int(index.ntotal) == 0:
            scores, positions = [], []
        else:
            fetch_k = (
                args.top_k
                if layer == "frames"
                else int(index.ntotal)
            )
            scores, positions = store.search(index, vector, fetch_k)
        latency_ms = (time.perf_counter() - query_started) * 1000.0

        matches = []
        if layer == "frames":
            for score, position in zip(scores, positions):
                if position < 0 or position >= len(records):
                    continue
                record = records[int(position)]
                matches.append(
                    {
                        "timestamp": float(record["start"]),
                        "score": float(score),
                        "path": record["path"],
                        "doc_id": record["doc_id"],
                    }
                )
        else:
            best_by_track: dict[int, dict[str, Any]] = {}
            for score, record in filter_object_matches(
                records, scores, positions, group["query"]
            ):
                meta = record.get("meta") or {}
                track_id = int(meta.get("track_id", 0))
                candidate = {
                    "timestamp": float(meta.get("timestamp_s", record["start"])),
                    "score": float(score),
                    "path": record["path"],
                    "doc_id": record["doc_id"],
                    "track_id": track_id,
                    "class_name": meta.get("class_name"),
                }
                existing = best_by_track.get(track_id)
                if existing is None or candidate["score"] > existing["score"]:
                    best_by_track[track_id] = candidate
            matches = sorted(
                best_by_track.values(), key=lambda item: item["score"], reverse=True
            )[: args.top_k]
        if not matches:
            results.append(
                {
                    "query_id": group["query_id"],
                    "video_id": group["video_id"],
                    "query": group["query"],
                    "type": group["type"],
                    "notes": group["notes"],
                    "intervals": group["intervals"],
                    "hits": {k: False for k in METRIC_KS},
                    "miou": 0.0,
                    "latency_ms": latency_ms,
                    "top1_timestamp": None,
                    "top1_score": 0.0,
                    "predicted_path": None,
                    "top5": [],
                }
            )
            _LOG.info(
                "[%2d/%2d] %-7s %-4s %-42s pred=    n/a gt=%s",
                number,
                len(groups),
                group["type"],
                "miss",
                group["query"][:42],
                ",".join(f"{start:.0f}-{end:.0f}" for start, end in group["intervals"]),
            )
            continue

        hits = {
            k: any(
                is_hit(match["timestamp"], group["intervals"], DEFAULT_TOLERANCE_S)
                for match in matches[:k]
            )
            for k in METRIC_KS
        }
        top1 = matches[0]
        miou = best_iou(top1["timestamp"], group["intervals"], DEFAULT_MIN_SEGMENT_S)

        results.append(
            {
                "query_id": group["query_id"],
                "video_id": group["video_id"],
                "query": group["query"],
                "type": group["type"],
                "notes": group["notes"],
                "intervals": group["intervals"],
                "hits": hits,
                "miou": miou,
                "latency_ms": latency_ms,
                "top1_timestamp": top1["timestamp"],
                "top1_score": top1["score"],
                "predicted_path": top1["path"],
                "top5": [
                    {"timestamp": match["timestamp"], "score": match["score"], "path": match["path"]}
                    for match in matches[:5]
                ],
            }
        )
        top1_mark = "HIT" if hits[1] else "miss"
        _LOG.info(
            "[%2d/%2d] %-7s %-4s %-42s pred=%7.1fs gt=%s",
            number,
            len(groups),
            group["type"],
            top1_mark,
            group["query"][:42],
            top1["timestamp"],
            ",".join(f"{start:.0f}-{end:.0f}" for start, end in group["intervals"]),
        )

    if not results:
        print("error: no results produced", file=sys.stderr)
        return 1

    per_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        per_type[result["type"]].append(result)
        per_video[result["video_id"]].append(result)

    failures = []
    for result in results:
        if result["hits"][1]:
            continue
        cause = _CAUSE_BY_TYPE.get(result["type"], "manual review needed")
        failures.append(
            {
                "query_id": result["query_id"],
                "video_id": result["video_id"],
                "type": result["type"],
                "query": result["query"],
                "intervals": result["intervals"],
                "top1_timestamp": result["top1_timestamp"],
                "top1_score": result["top1_score"],
                "predicted_path": result["predicted_path"],
                "cause": cause,
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": "A (frames)" if layer == "frames" else "B (objects)",
        "layer": layer,
        "model": model_name,
        "device": device,
        "queries_file": store.relative_path(queries_path, config.repo_root),
        "queries_sha256": sha256_file(queries_path),
        "query_count": len(results),
        "aggregate": summarize(results),
        "per_type": {key: summarize(value) for key, value in sorted(per_type.items())},
        "per_video": {key: summarize(value) for key, value in sorted(per_video.items())},
        "per_query": results,
        "failures": {
            "count": len(failures),
            "items": failures[: args.max_failures],
        },
    }

    store.write_json(out_path, payload)
    write_markdown(markdown_path, payload)

    gallery_path = None
    if not args.no_gallery and failures:
        gallery_path = build_gallery(failures, config, failures_dir, args.max_failures)
        if gallery_path is not None:
            store.write_json(
                failures_dir / "failures.json",
                {"generated_at": payload["generated_at"], "items": failures[: args.max_failures]},
            )
            _LOG.info("failure gallery: %s", gallery_path)

    elapsed = time.perf_counter() - started
    aggregate = payload["aggregate"]
    print()
    print(f"  {'Layer A baseline' if layer == 'frames' else 'Layer B object baseline'}")
    print(f"  Queries      : {aggregate['queries']}")
    print(f"  Recall@1     : {aggregate['recall@1']:.3f}")
    print(f"  Recall@5     : {aggregate['recall@5']:.3f}")
    print(f"  Recall@10    : {aggregate['recall@10']:.3f}")
    print(f"  mIoU (1 s)   : {aggregate['miou']:.3f}")
    print(f"  Latency p50  : {aggregate['latency_p50_ms']:.0f} ms")
    print(f"  Latency p95  : {aggregate['latency_p95_ms']:.0f} ms")
    print(f"  Results      : {out_path}")
    print(f"  Markdown     : {markdown_path}")
    if gallery_path is not None:
        print(f"  Gallery      : {gallery_path}")
    print(f"  Elapsed      : {format_duration(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
