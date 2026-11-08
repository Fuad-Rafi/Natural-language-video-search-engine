#!/usr/bin/env python
"""Index a video: probe metadata and extract timestamped frames.

Example:
    python scripts/index_video.py --video data/videos/sample.mp4
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
from src.index.video import extract_video, find_ffprobe, video_id_for  # noqa: E402
from src.utils.logging import format_duration, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index a video: probe it and extract timestamped frames."
    )
    parser.add_argument("--video", required=True, help="path to the source video")
    parser.add_argument(
        "--config",
        default=None,
        help="path to a YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="override the derived video id",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override video.sample_fps (default: 1.0)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: data/frames/<video_id>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild even when a cached index exists",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide progress bars and per-stage console logs",
    )
    return parser


def _audio_label(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    video_path = Path(args.video).expanduser()
    if not video_path.is_absolute():
        video_path = (Path.cwd() / video_path).resolve()
    else:
        video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 2

    video_id = args.video_id or video_id_for(video_path)
    output_dir = config.resolve(args.out) if args.out else config.frames_dir / video_id
    log_path = (
        config.logs_dir / f"index-video-{video_id}-{datetime.now():%Y%m%d-%H%M%S}.log"
    )

    logger = setup_logging(
        log_path,
        level=getattr(logging, args.log_level),
        console=not args.quiet,
    )
    logger.info("VideoSeek indexer starting")
    if find_ffprobe() is None:
        logger.warning(
            "ffprobe not found; falling back to OpenCV metadata. "
            "Install FFmpeg for exact frame counts and audio detection."
        )

    try:
        metadata, reused_cache = extract_video(
            video_path,
            config,
            video_id=video_id,
            sample_fps=args.fps,
            out_dir=output_dir,
            overwrite=args.overwrite,
            show_progress=not args.quiet,
        )
    except Exception:  # noqa: BLE001 - CLI boundary: report and continue
        logger.exception("indexing failed")
        print(f"\nIndexing failed. See log: {log_path}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    metadata_path = output_dir / "metadata.json"

    print()
    print(f"  Video        : {metadata['video_id']}")
    print(f"  Source       : {metadata['source_path']}")
    print(
        f"  Duration     : {format_duration(float(metadata['duration_s']))}"
        f" ({float(metadata['duration_s']):.2f} s)"
    )
    print(
        f"  Native       : {metadata['width']}x{metadata['height']} @ "
        f"{float(metadata['fps']):.3f} fps, "
        f"{metadata.get('frame_count_native') or '?'} frames, "
        f"{metadata.get('codec') or '?'}, "
        f"audio: {_audio_label(metadata.get('has_audio'))}"
    )
    print(
        f"  Sampled      : {float(metadata['sample_fps']):g} fps "
        f"(stride {metadata['stride']}) -> {metadata['frame_count']} frames"
    )
    print(f"  Output       : {output_dir}")
    print(f"  Metadata     : {metadata_path}")
    print(f"  Cache        : {'reused existing index' if reused_cache else 'built'}")
    print(f"  Log          : {log_path}")
    print(f"  Elapsed      : {format_duration(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
