"""Video probing and timestamped frame extraction (P0).

This module owns decoding only: it never computes embeddings. The output is a
directory of JPEG frames plus ``metadata.json`` / ``frames.jsonl`` that later
layers (frames, objects, OCR, captions) read instead of touching the source
video again.

Re-running the same video is cached: the source is hashed and an existing
index is reused when the hash and sample rate match.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2

from src.config import Config, REPO_ROOT
from src.utils.logging import StageTimer, get_logger

try:  # progress bars are nice but not required to index
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in the core requirements
    tqdm = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
JPEG_QUALITY = 95
_READ_CHUNK = 1024 * 1024
_LOG = get_logger("index.video")


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _winget_ffmpeg_bins() -> list[Path]:
    """Locate FFmpeg installed by winget, which does not refresh this process."""
    if os.name != "nt":
        return []
    package_roots = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        package_roots.append(Path(local) / "Microsoft" / "WinGet" / "Packages")
    package_roots.append(
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    )

    bins: list[Path] = []
    for root in package_roots:
        if root.exists():
            bins.extend(sorted(root.glob("Gyan.FFmpeg*/**/bin")))
    return bins


@functools.lru_cache(maxsize=1)
def find_ffprobe() -> Optional[Path]:
    which = shutil.which("ffprobe")
    if which:
        return Path(which)
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    for bin_dir in _winget_ffmpeg_bins():
        candidate = bin_dir / exe
        if candidate.exists():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> Optional[Path]:
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for bin_dir in _winget_ffmpeg_bins():
        candidate = bin_dir / exe
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def video_id_for(video_path: str | Path) -> str:
    """Derive a stable, filesystem-friendly id from the file name."""
    stem = Path(video_path).stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return (slug or "video")[:64]


def relative_path(path: str | Path, root: str | Path = REPO_ROOT) -> str:
    """Return a POSIX path relative to the repository when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: str | Path, chunk_size: int = _READ_CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate_to_float(value: Any) -> Optional[float]:
    if value in (None, "", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        num = _to_float(numerator)
        den = _to_float(denominator)
        if num is None or den in (None, 0.0):
            return None
        return num / den
    return _to_float(text)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _probe_with_ffprobe(ffprobe: Path, video_path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed ({result.returncode}): {result.stderr.strip()[:400]}"
        )

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream found in {video_path}")

    fmt = payload.get("format", {})
    fps = _rate_to_float(video.get("avg_frame_rate")) or _rate_to_float(
        video.get("r_frame_rate")
    )
    duration = _to_float(fmt.get("duration")) or _to_float(video.get("duration"))

    return {
        "duration_s": duration,
        "fps": fps,
        "width": _to_int(video.get("width")) or 0,
        "height": _to_int(video.get("height")) or 0,
        "frame_count_native": _to_int(video.get("nb_frames")),
        "codec": video.get("codec_name"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "bit_rate": _to_int(fmt.get("bit_rate")),
        "probe_backend": "ffprobe",
    }


def _probe_with_opencv(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None
        return {
            "duration_s": duration,
            "fps": fps if fps > 0 else None,
            "width": width,
            "height": height,
            "frame_count_native": frame_count or None,
            "codec": None,
            "has_audio": None,
            "bit_rate": None,
            "probe_backend": "opencv",
        }
    finally:
        capture.release()


def probe_video(video_path: str | Path) -> dict[str, Any]:
    """Return duration, fps, geometry and audio presence for a video."""
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    ffprobe = find_ffprobe()
    if ffprobe is not None:
        info = _probe_with_ffprobe(ffprobe, path)
    else:
        info = _probe_with_opencv(path)

    if not info.get("fps") or info["fps"] <= 0:
        raise RuntimeError(f"could not determine a valid fps for {path}")
    if not info.get("duration_s") and info.get("frame_count_native"):
        info["duration_s"] = info["frame_count_native"] / info["fps"]
    if not info.get("duration_s") or info["duration_s"] <= 0:
        raise RuntimeError(f"could not determine duration for {path}")
    if not info.get("width") or not info.get("height"):
        raise RuntimeError(f"could not determine frame size for {path}")

    info["source_path"] = relative_path(path)
    info["source_size_bytes"] = path.stat().st_size
    return info


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_frames(
    video_path: str | Path,
    out_dir: str | Path,
    sample_fps: float,
    *,
    native_fps: Optional[float] = None,
    native_frame_count: Optional[int] = None,
    jpeg_quality: int = JPEG_QUALITY,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], int, float]:
    """Decode the video and save one JPEG per sample interval.

    Frames are decoded sequentially and every ``stride``-th frame is kept, so
    each saved frame's timestamp is exact: ``source_frame_index / native_fps``.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    path = Path(video_path).expanduser().resolve()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = float(native_fps or capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"could not determine fps while sampling {path}")

    stride = max(1, int(round(fps / sample_fps)))
    total = native_frame_count
    if total is None and tqdm is not None:
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total = reported or None

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=total, unit="frame", desc="decode", dynamic_ncols=True)

    frames: list[dict[str, Any]] = []
    source_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_index % stride == 0:
                timestamp = source_index / fps
                frame_path = output / f"{len(frames):06d}.jpg"
                written = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not written:
                    raise RuntimeError(f"could not write frame: {frame_path}")

                height, width = frame.shape[:2]
                frames.append(
                    {
                        "index": len(frames),
                        "source_frame_index": source_index,
                        "timestamp_s": round(timestamp, 6),
                        "path": relative_path(frame_path),
                        "width": width,
                        "height": height,
                        "bytes": frame_path.stat().st_size,
                    }
                )

            source_index += 1
            if progress is not None:
                progress.update(1)
    finally:
        capture.release()
        if progress is not None:
            progress.close()

    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, stride, fps


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_frames_jsonl(path: Path, video_id: str, frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            record = {
                "doc_id": f"{video_id}:frame:{frame['index']:06d}",
                "video_id": video_id,
                "layer": "frame",
                "start": frame["timestamp_s"],
                "end": frame["timestamp_s"],
                "path": frame["path"],
                "text": None,
                "meta": {
                    "frame_idx": frame["index"],
                    "source_frame_idx": frame["source_frame_index"],
                    "timestamp_s": frame["timestamp_s"],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clear_frame_files(out_dir: Path) -> None:
    for path in out_dir.glob("*.jpg"):
        if re.fullmatch(r"\d{6}\.jpg", path.name):
            path.unlink()


def _cached_metadata(
    metadata_path: Path,
    source_hash: str,
    sample_fps: float,
    repo_root: Path,
) -> Optional[dict[str, Any]]:
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if existing.get("source_hash") != source_hash:
        return None
    try:
        cached_fps = float(existing.get("sample_fps"))
    except (TypeError, ValueError):
        return None
    if abs(cached_fps - sample_fps) > 1e-9:
        return None

    frames = existing.get("frames") or []
    if not frames:
        return None
    for frame in frames:
        candidate = Path(repo_root) / str(frame.get("path", ""))
        if not candidate.is_file():
            return None
    return existing


def extract_video(
    video_path: str | Path,
    config: Config,
    *,
    video_id: Optional[str] = None,
    sample_fps: Optional[float] = None,
    out_dir: Optional[str | Path] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Index one video and return ``(metadata, reused_cache)``."""
    path = Path(video_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    if sample_fps is None:
        sample_fps = float(config.get("video.sample_fps", 1.0))
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    video_id = video_id or video_id_for(path)
    output = config.resolve(out_dir) if out_dir else config.frames_dir / video_id
    metadata_path = output / "metadata.json"

    _LOG.info("source: %s", relative_path(path))
    _LOG.info("video_id: %s", video_id)

    with StageTimer("hash", _LOG):
        source_hash = f"sha256:{sha256_file(path)}"

    if not overwrite:
        cached = _cached_metadata(metadata_path, source_hash, sample_fps, config.repo_root)
        if cached is not None:
            _LOG.info(
                "cache hit: %s (%d frames @ %.3f fps)",
                video_id,
                int(cached.get("frame_count", len(cached.get("frames", [])))),
                sample_fps,
            )
            return cached, True
        if metadata_path.exists():
            _LOG.info("cache stale or incomplete; rebuilding %s", video_id)

    with StageTimer("probe", _LOG):
        probe = probe_video(path)

    _LOG.info(
        "native: %dx%d @ %.3f fps, %s frames, codec=%s, audio=%s",
        probe["width"],
        probe["height"],
        probe["fps"],
        probe.get("frame_count_native"),
        probe.get("codec"),
        probe.get("has_audio"),
    )
    if sample_fps > probe["fps"]:
        _LOG.warning(
            "sample_fps %.3f exceeds native fps %.3f; keeping every frame",
            sample_fps,
            probe["fps"],
        )

    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_frame_files(output)

    with StageTimer("frames", _LOG):
        frames, stride, used_fps = sample_frames(
            path,
            output,
            sample_fps,
            native_fps=probe["fps"],
            native_frame_count=probe.get("frame_count_native"),
            show_progress=show_progress,
        )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "videoseek",
        "video_id": video_id,
        "source_path": relative_path(path),
        "source_size_bytes": probe["source_size_bytes"],
        "source_hash": source_hash,
        "duration_s": round(float(probe["duration_s"]), 6),
        "fps": round(float(used_fps), 6),
        "width": probe["width"],
        "height": probe["height"],
        "frame_count_native": probe.get("frame_count_native"),
        "codec": probe.get("codec"),
        "has_audio": probe.get("has_audio"),
        "probe_backend": probe.get("probe_backend"),
        "sample_fps": float(sample_fps),
        "stride": stride,
        "frame_count": len(frames),
        "frames": frames,
        "config_snapshot": {
            key: config.get(key)
            for key in ("video", "frames")
            if config.get(key) is not None
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    with StageTimer("metadata", _LOG):
        _write_frames_jsonl(output / "frames.jsonl", video_id, frames)
        _write_json_atomic(metadata_path, metadata)

    _LOG.info("wrote %d frames to %s", len(frames), relative_path(output))
    return metadata, False
