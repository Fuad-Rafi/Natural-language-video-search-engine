#!/usr/bin/env python
"""Index a prepared public benchmark (frames, optionally objects).

Reads the query CSV to discover video ids, extracts 1 fps frames, and builds
Layer A. With ``--objects`` it also builds Layer B.

Example:
    python scripts/index_public.py --objects
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index.frames import index_frame_video  # noqa: E402
from src.index.objects import index_object_video  # noqa: E402
from src.index.video import extract_video  # noqa: E402
from src.utils.logging import format_duration, get_logger, setup_logging  # noqa: E402

_LOG = get_logger("index.public")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index the prepared Charades-STA subset."
    )
    parser.add_argument(
        "--queries", default="benchmarks/public/charades_sta_subset.csv"
    )
    parser.add_argument(
        "--videos", default="data/public/charades_sta/videos"
    )
    parser.add_argument("--objects", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(console=True)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    queries = config.resolve(args.queries)
    videos_dir = config.resolve(args.videos)
    if not queries.is_file():
        print(f"error: query CSV not found: {queries}", file=sys.stderr)
        return 2
    with queries.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    video_ids = sorted({row["video_id"] for row in rows})
    if args.limit:
        video_ids = video_ids[: args.limit]

    started = time.perf_counter()
    missing = 0
    indexed = 0
    for number, video_id in enumerate(video_ids, start=1):
        source = videos_dir / f"{video_id}.mp4"
        if not source.is_file():
            _LOG.warning("missing video: %s", source)
            missing += 1
            continue
        try:
            extract_video(
                source,
                config,
                video_id=video_id,
                overwrite=args.overwrite,
                show_progress=False,
            )
            index_frame_video(
                video_id,
                config,
                overwrite=args.overwrite,
                show_progress=False,
            )
            if args.objects:
                index_object_video(
                    video_id,
                    config,
                    overwrite=args.overwrite,
                    show_progress=False,
                )
            indexed += 1
        except Exception:  # noqa: BLE001 - continue the batch
            _LOG.exception("failed to index %s", video_id)
        if number % 10 == 0 or number == len(video_ids):
            print(
                f"[{number}/{len(video_ids)}] indexed={indexed} missing={missing}",
                flush=True,
            )

    print(
        f"\n  Videos indexed : {indexed}\n"
        f"  Missing        : {missing}\n"
        f"  Elapsed        : {format_duration(time.perf_counter() - started)}\n"
        f"  Frames dir     : {config.frames_dir}\n"
        f"  Indexes dir    : {config.indexes_dir}"
    )
    return 0 if indexed else 1


if __name__ == "__main__":
    raise SystemExit(main())
