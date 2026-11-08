#!/usr/bin/env python
"""Search indexed videos with a natural-language query (Layer A).

Examples:
    python scripts/search.py --query "a close-up of a human eye" --video sample
    python scripts/search.py --query "people in a control room" --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.search.query import (  # noqa: E402
    find_video_id,
    search_frames,
    search_objects,
    search_ocr,
    search_speech,
)
from src.utils.logging import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Natural-language search over indexed video.")
    parser.add_argument("--query", required=True, help="natural-language query")
    parser.add_argument(
        "--video",
        default=None,
        help="video id or path (default: search every indexed video)",
    )
    parser.add_argument(
        "--layer",
        default="frames",
        choices=["frames", "objects", "speech", "ocr"],
        help="retrieval layer to query (default: frames)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="number of results")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def format_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes, secs = divmod(total, 60.0)
    if minutes < 60:
        return f"{int(minutes):02d}:{secs:04.1f}"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{secs:04.1f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log_path = config.logs_dir / f"search-{datetime.now():%Y%m%d-%H%M%S}.log"
    logger = setup_logging(log_path, level=getattr(logging, args.log_level), console=False)

    video_id = find_video_id(args.video) if args.video else None
    search = {
        "frames": search_frames,
        "objects": search_objects,
        "speech": search_speech,
        "ocr": search_ocr,
    }[args.layer]
    try:
        payload = search(
            args.query,
            config,
            video_id=video_id,
            top_k=args.top_k,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and continue
        logger.exception("search failed")
        print(f"error: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print()
    print(f"  Query        : {payload['query']}")
    print(f"  Layer        : {payload['layer']}")
    print(f"  Model        : {payload['model']}")
    print(f"  Device       : {payload['device']}")
    print(f"  Videos       : {', '.join(payload['videos'])}")
    print(f"  Corpus       : {payload['total_records']} records")
    print()
    if payload["layer"] == "object":
        print(f"  {'#':>3}  {'Time':>9}  {'Score':>7}  {'Class':<12}  Crop")
        for rank, result in enumerate(payload["results"], start=1):
            print(
                f"  {rank:>3}  {format_timestamp(result['timestamp']):>9}  "
                f"{result['score']:>7.4f}  {str(result.get('class_name')):<12}  {result['path']}"
            )
    elif payload["layer"] in ("speech", "ocr"):
        label = "Transcript" if payload["layer"] == "speech" else "On-screen text"
        print(f"  {'#':>3}  {'Time':>9}  {'Score':>7}  {label}")
        for rank, result in enumerate(payload["results"], start=1):
            text = str(result.get("text") or "").replace("\n", " ")[:100]
            print(
                f"  {rank:>3}  {format_timestamp(result['timestamp']):>9}  "
                f"{result['score']:>7.4f}  {text}"
            )
    else:
        print(f"  {'#':>3}  {'Time':>9}  {'Score':>7}  Frame")
        for rank, result in enumerate(payload["results"], start=1):
            print(
                f"  {rank:>3}  {format_timestamp(result['timestamp']):>9}  "
                f"{result['score']:>7.4f}  {result['path']}"
            )
    print()
    print(f"  Log          : {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
