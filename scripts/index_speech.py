#!/usr/bin/env python
"""Build the Layer D speech index (Whisper transcript + bge/BM25 chunks).

Run after ``scripts/index_video.py`` has extracted frames (the frame metadata
provides the source hash and audio flag).

Example:
    python scripts/index_speech.py --video data/videos/lecture1.mp4
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
from src.index.speech import (  # noqa: E402
    audio_path,
    index_speech_video,
    speech_index_paths,
    transcript_path,
)
from src.index.video import video_id_for  # noqa: E402
from src.utils.logging import format_duration, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio and build the speech (Layer D) index."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="source video path (used to derive the video id)")
    source.add_argument("--video-id", help="id of an already-extracted video")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default=None, help="faster-whisper model (default: small)")
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
        / f"index-speech-{video_id}-{datetime.now():%Y%m%d-%H%M%S}.log"
    )
    logger = setup_logging(
        log_path,
        level=getattr(logging, args.log_level),
        console=not args.quiet,
    )
    logger.info("building Layer D (speech) index for %s", video_id)

    try:
        payload, reused_cache = index_speech_video(
            video_id,
            config,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            show_progress=not args.quiet,
        )
    except Exception:  # noqa: BLE001 - CLI boundary
        logger.exception("speech indexing failed")
        print(f"\nSpeech indexing failed. See log: {log_path}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    index_path, ids_path = speech_index_paths(config, video_id)
    stats = payload.get("stats") or {}

    print()
    print(f"  Video        : {video_id}")
    if payload.get("empty"):
        print(f"  Speech       : none ({payload.get('reason')})")
    else:
        print(f"  Segments     : {stats.get('segments')}")
        print(f"  Words        : {stats.get('words')}")
        print(f"  Chunks       : {stats.get('chunks')}")
        print(f"  Language     : {payload.get('language')}")
    print(f"  Whisper      : {payload.get('model')}")
    print(f"  Embeddings   : {payload.get('embedding_model')} ({payload.get('dim')}-d)")
    print(f"  Device       : {payload.get('device')}")
    print(f"  Audio        : {audio_path(config, video_id)}")
    print(f"  Transcript   : {transcript_path(config, video_id)}")
    print(f"  FAISS        : {index_path}")
    print(f"  Sidecar      : {ids_path}")
    print(f"  Cache        : {'reused existing index' if reused_cache else 'built'}")
    print(f"  Log          : {log_path}")
    print(f"  Elapsed      : {format_duration(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
