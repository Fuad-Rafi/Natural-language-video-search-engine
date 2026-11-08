#!/usr/bin/env python
"""Verify extracted frames against the source video.

This is the P0 exit-criterion check:
  * metadata and frame files are complete and consistent
  * timestamps equal ``source_frame_index / fps``
  * sampled frames re-decode to the same pixels as the saved JPEGs
  * a contact sheet of the sampled frames is produced for a human spot-check

Example:
    python scripts/verify_frames.py --video data/videos/sample.mp4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index.video import relative_path, video_id_for  # noqa: E402

FRAME_NAME = re.compile(r"\d{6}\.jpg$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify extracted frames and timestamps against the source video."
    )
    parser.add_argument("--frames-dir", default=None, help="e.g. data/frames/sample")
    parser.add_argument("--video", default=None, help="source video (used to find defaults)")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--samples", type=int, default=5, help="frames to re-decode")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--max-diff",
        type=float,
        default=8.0,
        help="max mean absolute pixel difference after JPEG re-encode",
    )
    parser.add_argument(
        "--max-timestamp-error",
        type=float,
        default=0.02,
        help="max |timestamp - source_frame_index / fps| in seconds",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="skip writing the visual contact sheet",
    )
    return parser


def resolve_record_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def mean_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[:2] != second.shape[:2]:
        second = cv2.resize(
            second,
            (first.shape[1], first.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if first.ndim == 3 else first
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY) if second.ndim == 3 else second
    return float(np.mean(cv2.absdiff(first_gray, second_gray)))


def make_contact_sheet(
    frames: list[dict[str, Any]],
    diffs: dict[int, float],
    repo_root: Path,
    out_path: Path,
    tile_height: int = 240,
) -> Path | None:
    tiles = []
    for frame in frames:
        stored = cv2.imread(
            str(resolve_record_path(repo_root, frame["path"])),
            cv2.IMREAD_COLOR,
        )
        if stored is None:
            continue
        scale = tile_height / stored.shape[0]
        tile = cv2.resize(
            stored,
            (max(1, int(stored.shape[1] * scale)), tile_height),
            interpolation=cv2.INTER_AREA,
        )
        label = f"t={float(frame['timestamp_s']):.1f}s  #{frame['index']}"
        if frame["index"] in diffs:
            label += f"  diff={diffs[frame['index']]:.2f}"
        cv2.rectangle(tile, (0, tile_height - 28), (tile.shape[1], tile_height), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (8, tile_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return None
    sheet = np.hstack(tiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        return None
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        print("error: --samples must be at least 1", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.frames_dir:
        frames_dir = config.resolve(args.frames_dir)
    elif args.video:
        frames_dir = config.frames_dir / video_id_for(args.video)
    else:
        print("error: pass --video or --frames-dir", file=sys.stderr)
        return 2

    metadata_path = frames_dir / "metadata.json"
    if not metadata_path.is_file():
        print(f"error: metadata not found: {metadata_path}", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = metadata.get("frames") or []

    video_path = (
        config.resolve(args.video)
        if args.video
        else resolve_record_path(config.repo_root, str(metadata.get("source_path", "")))
    )

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"Verifying {frames_dir}")
    print(f"Source   {video_path}")
    print()

    count_ok = metadata.get("schema_version") == 1 and bool(frames)
    count_ok = count_ok and metadata.get("frame_count") == len(frames)
    record(
        "metadata",
        count_ok,
        f"schema={metadata.get('schema_version')} frames={len(frames)} "
        f"(declared {metadata.get('frame_count')})",
    )

    missing = []
    empty = []
    for frame in frames:
        path = resolve_record_path(config.repo_root, frame["path"])
        if not path.is_file():
            missing.append(frame["path"])
        elif path.stat().st_size == 0:
            empty.append(frame["path"])
    expected_names = {Path(frame["path"]).name for frame in frames}
    actual_names = {
        path.name for path in frames_dir.glob("*.jpg") if FRAME_NAME.fullmatch(path.name)
    }
    orphans = sorted(actual_names - expected_names)
    record(
        "frame files",
        bool(frames) and not missing and not empty and not orphans,
        f"missing={len(missing)} empty={len(empty)} orphan={len(orphans)}",
    )

    fps = float(metadata.get("fps") or 0.0)
    if fps <= 0:
        record("timestamp mapping", False, "metadata has no valid fps")
    else:
        errors = [
            abs(float(frame["timestamp_s"]) - frame["source_frame_index"] / fps)
            for frame in frames
        ]
        max_error = max(errors) if errors else float("inf")
        increasing = all(
            later["timestamp_s"] > earlier["timestamp_s"]
            for earlier, later in zip(frames, frames[1:])
        )
        record(
            "timestamp mapping",
            bool(errors) and max_error <= args.max_timestamp_error and increasing,
            f"max |ts - idx/fps| = {max_error:.6f}s, strictly increasing={increasing}",
        )

    duration = float(metadata.get("duration_s") or 0.0)
    if frames:
        first = float(frames[0]["timestamp_s"])
        last = float(frames[-1]["timestamp_s"])
        tail_gap = duration - last
        record(
            "coverage",
            first <= 0.5 and 0.0 <= tail_gap <= 2.0,
            f"first={first:.3f}s last={last:.3f}s duration={duration:.3f}s "
            f"tail_gap={tail_gap:.3f}s",
        )
    else:
        record("coverage", False, "no frames to check")

    if not video_path.is_file():
        record("source video", False, f"not found: {video_path}")
        sampled: list[dict[str, Any]] = []
        diffs: dict[int, float] = {}
    else:
        record("source video", True, relative_path(video_path, config.repo_root))
        rng = random.Random(args.seed)
        sampled = sorted(
            rng.sample(frames, min(args.samples, len(frames))),
            key=lambda frame: frame["source_frame_index"],
        )
        targets = {frame["source_frame_index"]: frame for frame in sampled}
        diffs = {}
        unreadable: list[str] = []

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            record("re-decode", False, f"OpenCV could not open {video_path}")
        else:
            source_index = 0
            try:
                while targets:
                    ok, decoded = capture.read()
                    if not ok:
                        break
                    frame = targets.get(source_index)
                    if frame is not None:
                        stored = cv2.imread(
                            str(resolve_record_path(config.repo_root, frame["path"])),
                            cv2.IMREAD_COLOR,
                        )
                        if stored is None:
                            unreadable.append(frame["path"])
                        else:
                            diffs[frame["index"]] = mean_abs_diff(stored, decoded)
                        targets.pop(source_index)
                    source_index += 1
            finally:
                capture.release()

            if targets:
                record(
                    "re-decode",
                    False,
                    f"{len(targets)} sampled source frames were never reached",
                )
            elif unreadable:
                record(
                    "re-decode",
                    False,
                    f"{len(unreadable)} saved JPEGs could not be read",
                )
            elif not diffs:
                record("re-decode", False, "no frames compared")
            else:
                worst = max(diffs.values())
                record(
                    "re-decode",
                    worst <= args.max_diff,
                    f"{len(diffs)} frames re-decoded, max mean |diff| = {worst:.2f} "
                    f"(limit {args.max_diff})",
                )

    contact_sheet = None
    if not args.no_contact_sheet and sampled:
        contact_sheet = make_contact_sheet(
            sampled,
            diffs,
            config.repo_root,
            frames_dir / "verify_contact_sheet.jpg",
        )
        if contact_sheet is not None:
            print(f"[INFO] contact sheet: {contact_sheet}")
        else:
            print("[INFO] contact sheet could not be written")

    passed = all(check["ok"] for check in checks)
    report = {
        "video_id": metadata.get("video_id"),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "samples": [
            {
                "index": frame["index"],
                "timestamp_s": frame["timestamp_s"],
                "source_frame_index": frame["source_frame_index"],
                "mean_abs_diff": diffs.get(frame["index"]),
            }
            for frame in sampled
        ],
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
    }
    (frames_dir / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    if passed:
        print(f"PASS: {len(checks)} checks passed. Report: {frames_dir / 'verification.json'}")
        return 0
    failed = [check["name"] for check in checks if not check["ok"]]
    print(f"FAIL: {len(failed)} check(s) failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
