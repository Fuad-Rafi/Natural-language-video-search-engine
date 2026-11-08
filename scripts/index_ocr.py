#!/usr/bin/env python
"""Build the Layer E OCR index (keyframes -> PaddleOCR -> bge/BM25).

Run after ``scripts/index_video.py`` has extracted frames.

Example:
    python scripts/index_ocr.py --video data/videos/lecture1.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index.ocr import index_ocr_video, ocr_index_paths, ocr_path  # noqa: E402
from src.index.video import video_id_for  # noqa: E402
from src.utils.logging import format_duration, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on keyframes and build the OCR (Layer E) index."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="source video path (used to derive the video id)")
    source.add_argument("--video-id", help="id of an already-extracted video")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    video_id = args.video_id or video_id_for(args.video)
    metadata_path = config.frames_dir / video_id / "metadata.json"
    if not metadata_path.is_file():
        print(
            f"error: no extracted frames for {video_id!r} ({metadata_path}).\n"
            f"Run: python scripts/index_video.py --video {args.video or '<video>'}",
            file=sys.stderr,
        )
        return 2

    log_path = (
        config.logs_dir
        / f"index-ocr-{video_id}-{datetime.now():%Y%m%d-%H%M%S}.log"
    )
    logger = setup_logging(
        log_path,
        level=getattr(logging, args.log_level),
        console=not args.quiet,
    )
    logger.info("building Layer E (OCR) index for %s", video_id)

    try:
        payload, reused_cache = index_ocr_video(
            video_id,
            config,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            show_progress=not args.quiet,
        )
    except Exception:  # noqa: BLE001 - CLI boundary
        logger.exception("OCR indexing failed")
        print(f"\nOCR indexing failed. See log: {log_path}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    index_path, ids_path = ocr_index_paths(config, video_id)
    stats = payload.get("stats") or {}

    print()
    print(f"  Video        : {video_id}")
    if payload.get("empty"):
        print(f"  OCR          : none ({payload.get('reason')})")
    else:
        print(f"  Keyframes    : {stats.get('keyframes')}")
        print(f"  Segments     : {stats.get('segments')}")
        print(f"  Lines        : {stats.get('lines')}")
    print(f"  Language     : {payload.get('lang')}")
    print(f"  Engine       : {payload.get('engine')}")
    print(f"  Embeddings   : {payload.get('embedding_model')} ({payload.get('dim')}-d)")
    print(f"  Device       : {payload.get('device')}")
    print(f"  OCR JSON     : {ocr_path(config, video_id)}")
    print(f"  FAISS        : {index_path}")
    print(f"  Sidecar      : {ids_path}")
    print(f"  Cache        : {'reused existing index' if reused_cache else 'built'}")
    print(f"  Log          : {log_path}")
    print(f"  Elapsed      : {format_duration(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
