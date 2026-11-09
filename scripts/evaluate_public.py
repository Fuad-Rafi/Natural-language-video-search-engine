#!/usr/bin/env python
"""Evaluate Layer A on a Charades-STA subset with a fixed-window proposal.

Charades-STA clips are ~30 s and the train-split median annotation length is
7.4 s. Frame retrieval returns a peak frame; this script turns it into a
``--window-s`` proposal centred on that frame (default 7.5 s) and reports the
standard R@1 IoU 0.5/0.7 and mIoU metrics.

Example:
    python scripts/evaluate_public.py --window-s 7.5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index import store  # noqa: E402
from src.search.query import load_frame_index  # noqa: E402
from src.utils.logging import format_duration, get_logger, setup_logging  # noqa: E402

_LOG = get_logger("evaluate.public")
WINDOWS = (4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 10.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Layer A on a Charades-STA subset."
    )
    parser.add_argument(
        "--queries", default="benchmarks/public/charades_sta_subset.csv"
    )
    parser.add_argument("--window-s", type=float, default=7.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--out",
        default="benchmarks/public/results/charades_sta_a_window75.json",
    )
    parser.add_argument(
        "--markdown",
        default="benchmarks/public/results/charades_sta_a_window75.md",
    )
    parser.add_argument("--config", default=None)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "query_id": raw["query_id"],
                    "video_id": raw["video_id"],
                    "query": raw["query"],
                    "start": float(raw["start"]),
                    "end": float(raw["end"]),
                }
            )
    return rows


def temporal_iou(
    pred_start: float, pred_end: float, gt_start: float, gt_end: float
) -> float:
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def window_for(
    peak: float, window_s: float, duration: float
) -> tuple[float, float]:
    half = window_s / 2.0
    start = max(0.0, peak - half)
    end = min(duration, start + window_s)
    start = max(0.0, end - window_s)
    return start, end


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    setup_logging(console=True)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from src.index.frames import embed_texts, load_siglip, resolve_device

    queries_path = config.resolve(args.queries)
    rows = read_queries(queries_path)
    video_ids = sorted({row["video_id"] for row in rows})
    device = resolve_device(config.get("frames.device", "auto"))
    model_name = str(config.get("frames.model"))
    _LOG.info("loading %s on %s", model_name, device)
    processor, model = load_siglip(model_name, device)

    frame_indexes = {}
    durations = {}
    for video_id in video_ids:
        frame_indexes[video_id] = load_frame_index(config, video_id)
        metadata_path = config.frames_dir / video_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        durations[video_id] = float(metadata["duration_s"])

    results = []
    peaks: list[float] = []
    for number, row in enumerate(rows, start=1):
        vector = embed_texts([row["query"]], processor, model, device=device)[0]
        index, payload = frame_indexes[row["video_id"]]
        scores, positions = store.search(
            index, vector, min(args.top_k, int(index.ntotal))
        )
        records = payload["records"]
        peak = None
        for position in positions:
            position = int(position)
            if 0 <= position < len(records):
                peak = float(records[position]["start"])
                break
        if peak is None:
            peak = 0.0
        peaks.append(peak)
        start, end = window_for(peak, args.window_s, durations[row["video_id"]])
        iou = temporal_iou(start, end, row["start"], row["end"])
        results.append(
            {
                "query_id": row["query_id"],
                "video_id": row["video_id"],
                "query": row["query"],
                "gt_start": row["start"],
                "gt_end": row["end"],
                "peak": round(peak, 3),
                "pred_start": round(start, 3),
                "pred_end": round(end, 3),
                "iou": round(iou, 4),
                "any_overlap": bool(
                    min(end, row["end"]) > max(start, row["start"])
                ),
            }
        )
        if number % 50 == 0 or number == len(rows):
            _LOG.info("[%d/%d] top1 IoU mean %.3f", number, len(rows), np.mean([r["iou"] for r in results]))

    def aggregate(window_s: float | None = None) -> dict[str, Any]:
        values = results
        if window_s is not None:
            values = []
            for row in results:
                start, end = window_for(
                    row["peak"], window_s, durations[row["video_id"]]
                )
                iou = temporal_iou(start, end, row["gt_start"], row["gt_end"])
                values.append(
                    {
                        "iou": iou,
                        "any_overlap": bool(
                            min(end, row["gt_end"]) > max(start, row["gt_start"])
                        ),
                    }
                )
        count = len(values)
        return {
            "queries": count,
            "recall@1_iou_0_5": round(
                sum(1 for row in values if row["iou"] >= 0.5) / count, 4
            ),
            "recall@1_iou_0_7": round(
                sum(1 for row in values if row["iou"] >= 0.7) / count, 4
            ),
            "miou": round(sum(row["iou"] for row in values) / count, 4),
            "any_overlap": round(
                sum(1 for row in values if row["any_overlap"]) / count, 4
            ),
        }

    aggregate_main = aggregate()
    sensitivity = {str(window): aggregate(window) for window in WINDOWS}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": "Charades-STA test subset",
        "model": model_name,
        "device": device,
        "queries_file": store.relative_path(queries_path, config.repo_root),
        "queries_sha256": sha256_file(queries_path),
        "videos": len(video_ids),
        "queries": len(results),
        "window_s": args.window_s,
        "protocol": (
            "Frame retrieval (Layer A) at 1 fps; top-1 frame becomes a fixed "
            f"{args.window_s:.1f}s window centred on the peak. Window chosen "
            "from the Charades-STA train median duration (7.4s), not tuned on test."
        ),
        "aggregate": aggregate_main,
        "window_sensitivity": sensitivity,
        "per_query": results,
    }
    out_path = config.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_path = config.resolve(args.markdown)
    lines = [
        "# Charades-STA test subset — Layer A",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Model: `{model_name}`",
        f"- Videos: {len(video_ids)} · Queries: {len(results)}",
        f"- Proposal: {args.window_s:.1f}s window centred on the top-1 frame",
        f"- Queries SHA-256: `{payload['queries_sha256'][:16]}…`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| R@1 IoU 0.5 | {aggregate_main['recall@1_iou_0_5']:.3f} |",
        f"| R@1 IoU 0.7 | {aggregate_main['recall@1_iou_0_7']:.3f} |",
        f"| mIoU | {aggregate_main['miou']:.3f} |",
        f"| Any overlap | {aggregate_main['any_overlap']:.3f} |",
        "",
        "## Window sensitivity",
        "",
        "| Window (s) | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |",
        "|---|---|---|---|",
    ]
    for window, summary in sensitivity.items():
        lines.append(
            f"| {window} | {summary['recall@1_iou_0_5']:.3f} | "
            f"{summary['recall@1_iou_0_7']:.3f} | {summary['miou']:.3f} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"  Benchmark       : Charades-STA test subset")
    print(f"  Videos          : {len(video_ids)}")
    print(f"  Queries         : {len(results)}")
    print(f"  Window          : {args.window_s:.1f} s")
    print(f"  R@1 IoU 0.5     : {aggregate_main['recall@1_iou_0_5']:.3f}")
    print(f"  R@1 IoU 0.7     : {aggregate_main['recall@1_iou_0_7']:.3f}")
    print(f"  mIoU            : {aggregate_main['miou']:.3f}")
    print(f"  Any overlap     : {aggregate_main['any_overlap']:.3f}")
    print(f"  Results         : {out_path}")
    print(f"  Markdown        : {md_path}")
    print(f"  Elapsed         : {format_duration(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
