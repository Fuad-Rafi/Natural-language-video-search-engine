#!/usr/bin/env python
"""Generate a <60 s demo montage from a few search results.

Each query gets a title card followed by its top clip, normalised to 720p/30fps
with AAC audio so the pieces can be concatenated. The output is gitignored.

Example:
    python scripts/make_demo.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.index.video import find_ffmpeg  # noqa: E402
from src.search import service  # noqa: E402

WIDTH, HEIGHT = 1280, 720
FPS = 30
FONT_PATHS = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def title_card(query: str, segment: dict, path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (17, 24, 39))
    draw = ImageDraw.Draw(image)
    small = _font(28)
    big = _font(54)
    draw.text((70, 60), "VideoSeek", font=_font(34), fill=(96, 165, 250))
    draw.text((70, 120), "natural-language search over long video", font=small, fill=(156, 163, 175))

    lines = _wrap(query, big, WIDTH - 140)
    y = 230
    for line in lines[:4]:
        draw.text((70, y), line, font=big, fill=(249, 250, 251))
        y += 68

    badges = " · ".join(segment.get("badges", []))
    timestamp = f"{segment['start']:.1f}s – {segment['end']:.1f}s"
    draw.text((70, 560), f"{badges}   {timestamp}", font=small, fill=(250, 204, 21))
    draw.text((70, 620), "timestamped segment · source badge · playable clip", font=small, fill=(156, 163, 175))
    image.save(path)


def run(command: list[str]) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "")[:600])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a VideoSeek demo montage.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "where he says young people who break the law",
            "the command sfc /scannow in command prompt",
            "a man making espresso",
        ],
    )
    parser.add_argument("--video-ids", nargs="+", default=["lecture1", "screen-recording", "indoor1"])
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--out", default="docs/demo.mp4")
    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2

    temp = config.data_dir / "demo_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []

    for index, (query, video_id) in enumerate(zip(args.queries, args.video_ids), start=1):
        result = service.search(
            query, config, video_id=video_id, layers="auto", top_k=1
        )
        if not result["segments"]:
            print(f"skip (no result): {query}")
            continue
        segment = result["segments"][0]
        source = service.resolve_source(config, video_id)
        if source is None:
            print(f"skip (no source): {video_id}")
            continue

        card = temp / f"{index:02d}_title.png"
        title_card(query, segment, card)
        title_mp4 = temp / f"{index:02d}_title.mp4"
        run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-i", str(card),
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                "-t", "2.5",
                "-vf", f"scale={WIDTH}:{HEIGHT},format=yuv420p",
                "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest",
                str(title_mp4),
            ]
        )
        parts.append(title_mp4)

        start = max(0.0, float(segment["start"]) - 1.0)
        duration = min(args.clip_seconds, max(2.0, float(segment["end"]) - start + 1.0))
        clip_mp4 = temp / f"{index:02d}_clip.mp4"
        has_audio = bool(
            service._video_metadata(config, video_id).get("has_audio")
        )
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(source),
        ]
        if has_audio:
            audio_map = ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
        else:
            command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            audio_map = ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
        command += [
            "-t", f"{duration:.3f}",
            "-vf",
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p",
            "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            *audio_map, "-shortest", str(clip_mp4),
        ]
        run(command)
        parts.append(clip_mp4)
        print(f"[{index}] {query} -> {segment['start']:.1f}s {segment['badges']}")

    if not parts:
        print("error: no parts generated", file=sys.stderr)
        return 1

    list_file = temp / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{part.as_posix()}'" for part in parts) + "\n",
        encoding="utf-8",
    )
    out_path = config.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]
    )
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
