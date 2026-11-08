"""Layer D: audio extraction, Whisper transcription, and the speech index.

The transcript is the expensive artifact, so it is cached separately from the
FAISS index: changing the text embedding model re-embeds the chunks without
transcribing the audio again. The audio itself is 16 kHz mono PCM WAV so
Whisper timestamps line up with the video timeline.

Text chunks are embedded with ``bge-small`` and (at query time) scored with
BM25 as well. Search combines the two rank lists with RRF.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import Config
from src.index import store
from src.index.frames import resolve_device
from src.index.text import DEFAULT_TEXT_MODEL, encode_texts, load_text_encoder
from src.index.video import find_ffmpeg
from src.utils.logging import StageTimer, get_logger

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in the core requirements
    tqdm = None  # type: ignore[assignment]

_LOG = get_logger("index.speech")

SCHEMA_VERSION = 1
DEFAULT_WHISPER_MODEL = "small"
SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 30.0
DEFAULT_OVERLAP_S = 5.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def audio_path(config: Config, video_id: str) -> Path:
    return config.audio_dir / f"{video_id}.wav"


def transcript_path(config: Config, video_id: str) -> Path:
    return config.transcripts_dir / f"{video_id}.json"


def speech_index_paths(config: Config, video_id: str) -> tuple[Path, Path]:
    base = config.indexes_dir / video_id
    return base / "speech.faiss", base / "speech.ids.json"


def _read_metadata(config: Config, video_id: str) -> dict[str, Any]:
    metadata_path = config.frames_dir / video_id / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"frame metadata not found: {metadata_path}. "
            f"Run scripts/index_video.py --video <video> first."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not payload.get("frames"):
        raise ValueError(f"no frames listed in {metadata_path}")
    return payload


# ---------------------------------------------------------------------------
# CUDA DLL discovery (Windows / faster-whisper)
# ---------------------------------------------------------------------------


def _prepare_cuda_dll_path() -> None:
    """Add pip-installed NVIDIA DLL directories to the process search path.

    ``faster-whisper`` loads cuBLAS/cuDNN through CTranslate2. On Windows those
    DLLs live under ``site-packages/nvidia/<package>/bin`` and are not on PATH
    by default (challenge C-21).
    """
    if os.name != "nt":
        return
    roots: list[str] = []
    try:
        import site

        roots.extend(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
    except Exception:  # pragma: no cover - site is expected to work
        pass
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))

    added: set[str] = set()
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            directory = str(bin_dir)
            if directory in added:
                continue
            added.add(directory)
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(directory)
            except (AttributeError, OSError):
                pass
    if added:
        _LOG.debug("registered %d NVIDIA DLL directories", len(added))


def compute_type_for(device: str, preference: Optional[str]) -> str:
    """Map ``auto`` to float16 on CUDA and int8 on CPU."""
    chosen = (preference or "auto").lower()
    if chosen != "auto":
        return chosen
    return "float16" if device == "cuda" else "int8"


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(
    video_source: str | Path,
    output: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    overwrite: bool = False,
) -> Optional[Path]:
    """Extract mono PCM WAV audio, or return ``None`` when there is no audio."""
    source = Path(video_source)
    destination = Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"source video not found: {source}")
    if destination.is_file() and not overwrite:
        return destination

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install FFmpeg to extract audio"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        lowered = stderr.lower()
        if any(
            marker in lowered
            for marker in (
                "does not contain any stream",
                "matches no streams",
                "could not find codec parameters",
                "output file #0 does not contain any stream",
            )
        ):
            _LOG.warning("no audio stream in %s; skipping speech layer", source)
            destination.unlink(missing_ok=True)
            return None
        raise RuntimeError(
            f"ffmpeg audio extraction failed ({result.returncode}): {stderr[:400]}"
        )

    if not destination.is_file() or destination.stat().st_size <= 44:
        destination.unlink(missing_ok=True)
        _LOG.warning("ffmpeg produced no audio for %s", source)
        return None
    return destination


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_file: str | Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str = "auto",
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Transcribe a WAV file with faster-whisper and return a JSON payload."""
    _prepare_cuda_dll_path()
    from faster_whisper import WhisperModel

    source = Path(audio_file)
    if not source.is_file():
        raise FileNotFoundError(f"audio file not found: {source}")

    resolved = resolve_device(device)
    chosen_compute = compute_type_for(resolved, compute_type)
    _LOG.info(
        "loading Whisper %s on %s (%s)", model_name, resolved, chosen_compute
    )
    model = WhisperModel(model_name, device=resolved, compute_type=chosen_compute)

    segments_iter, info = model.transcribe(
        str(source),
        language=language or None,
        beam_size=max(1, int(beam_size)),
        word_timestamps=bool(word_timestamps),
        vad_filter=bool(vad_filter),
    )

    progress = None
    total = float(getattr(info, "duration", 0.0) or 0.0)
    if show_progress and tqdm is not None and total > 0:
        progress = tqdm(total=total, unit="s", desc="transcribe", dynamic_ncols=True)

    segments: list[dict[str, Any]] = []
    try:
        for segment in segments_iter:
            words = []
            for word in getattr(segment, "words", None) or []:
                probability = getattr(word, "probability", None)
                words.append(
                    {
                        "w": str(word.word),
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "prob": None
                        if probability is None
                        else round(float(probability), 4),
                    }
                )
            segments.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": str(segment.text).strip(),
                    "words": words,
                }
            )
            if progress is not None:
                progress.update(float(segment.end) - float(segment.start))
    finally:
        if progress is not None:
            progress.close()

    return {
        "model": model_name,
        "device": resolved,
        "compute_type": chosen_compute,
        "language": getattr(info, "language", None),
        "language_probability": _round_or_none(
            getattr(info, "language_probability", None), 4
        ),
        "duration_s": _round_or_none(getattr(info, "duration", None), 3),
        "duration_after_vad_s": _round_or_none(
            getattr(info, "duration_after_vad", None), 3
        ),
        "segments": segments,
    }


def _round_or_none(value: Any, digits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_transcript(
    segments: list[dict[str, Any]],
    *,
    chunk_seconds: float = DEFAULT_CHUNK_S,
    overlap_seconds: float = DEFAULT_OVERLAP_S,
) -> list[dict[str, Any]]:
    """Turn transcript segments into overlapping, word-aligned chunks."""
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")

    words: list[dict[str, Any]] = []
    for segment in segments:
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                text = str(word.get("w") or "").strip()
                if not text:
                    continue
                words.append(
                    {
                        "w": text,
                        "start": float(word.get("start", segment.get("start", 0.0))),
                        "end": float(word.get("end", segment.get("end", 0.0))),
                    }
                )
        else:
            text = str(segment.get("text") or "").strip()
            if text:
                words.append(
                    {
                        "w": text,
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                    }
                )

    if not words:
        return []
    words.sort(key=lambda item: item["start"])

    overlap = max(0.0, float(overlap_seconds))
    chunks: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(words):
        start = words[start_index]["start"]
        end_index = start_index
        while (
            end_index + 1 < len(words)
            and words[end_index]["end"] - start < chunk_seconds
        ):
            end_index += 1

        chunk_words = words[start_index : end_index + 1]
        end = max(chunk_words[-1]["end"], start + 0.001)
        text = " ".join(item["w"] for item in chunk_words).strip()
        if text:
            chunks.append(
                {
                    "index": len(chunks),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "words": chunk_words,
                }
            )

        if end_index >= len(words) - 1:
            break

        target = end - overlap
        next_index = start_index + 1
        while next_index <= end_index and words[next_index]["start"] < target:
            next_index += 1
        if next_index <= start_index:
            next_index = start_index + 1
        start_index = next_index
    return chunks


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _settings(config: Config, model_name: str, embedding_model: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "language": config.get("speech.language"),
        "beam_size": int(config.get("speech.beam_size", 5)),
        "word_timestamps": bool(config.get("speech.word_timestamps", True)),
        "vad_filter": bool(config.get("speech.vad_filter", True)),
        "chunk_seconds": float(
            config.get("speech.chunk_seconds", DEFAULT_CHUNK_S)
        ),
        "overlap_seconds": float(
            config.get("speech.overlap_seconds", DEFAULT_OVERLAP_S)
        ),
        "embedding_model": embedding_model,
    }


def _transcript_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: settings[key]
        for key in (
            "model",
            "language",
            "beam_size",
            "word_timestamps",
            "vad_filter",
        )
    }


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _transcript_is_cached(
    payload: Optional[dict[str, Any]],
    metadata: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    if not payload:
        return False
    return (
        payload.get("source_hash") == metadata.get("source_hash")
        and payload.get("settings") == _transcript_settings(settings)
        and bool(payload.get("segments"))
    )


def _cached_index(
    ids_path: Path,
    index_path: Path,
    metadata: dict[str, Any],
    settings: dict[str, Any],
) -> Optional[dict[str, Any]]:
    payload = _load_json(ids_path)
    if not payload:
        return None
    if payload.get("settings") != settings:
        return None
    if payload.get("source_hash") != metadata.get("source_hash"):
        return None
    if payload.get("empty"):
        return payload
    if not index_path.is_file():
        return None
    if int(payload.get("count", -1)) != len(payload.get("records") or []):
        return None
    return payload


def _write_empty_sidecar(
    video_id: str,
    config: Config,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    index_path, ids_path = speech_index_paths(config, video_id)
    index_path.unlink(missing_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "layer": "speech",
        "model": settings["model"],
        "embedding_model": settings["embedding_model"],
        "device": None,
        "dim": None,
        "normalized": True,
        "count": 0,
        "empty": True,
        "reason": reason,
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "audio_path": store.relative_path(audio_path(config, video_id), config.repo_root),
        "transcript_path": store.relative_path(
            transcript_path(config, video_id), config.repo_root
        ),
        "settings": settings,
        "stats": {"segments": 0, "words": 0, "chunks": 0},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": [],
    }
    store.write_json(ids_path, payload)
    _LOG.warning("%s has no speech (%s); wrote an empty speech sidecar", video_id, reason)
    return payload


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def index_speech_video(
    video_id: str,
    config: Config,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Build (or reuse) the Layer D index for one video.

    Returns ``(sidecar_payload, reused_cache)``.
    """
    metadata = _read_metadata(config, video_id)
    whisper_model = str(model_name or config.get("speech.model") or DEFAULT_WHISPER_MODEL)
    embedding_model = str(
        config.get("text.embedding_model") or DEFAULT_TEXT_MODEL
    )
    batch_size = int(batch_size or config.get("text.batch_size", 32))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    settings = _settings(config, whisper_model, embedding_model)
    index_path, ids_path = speech_index_paths(config, video_id)

    if not overwrite:
        cached = _cached_index(ids_path, index_path, metadata, settings)
        if cached is not None:
            _LOG.info(
                "cache hit: %s speech index (%d chunks, %s)",
                video_id,
                int(cached.get("count", 0)),
                embedding_model,
            )
            return cached, True
        if index_path.exists() or ids_path.exists():
            _LOG.info("speech index stale; rebuilding %s", video_id)

    if metadata.get("has_audio") is False:
        return (
            _write_empty_sidecar(
                video_id, config, metadata, settings, reason="no_audio_stream"
            ),
            False,
        )

    transcript_file = transcript_path(config, video_id)
    transcript = _load_json(transcript_file)
    transcript_cached = (
        not overwrite
        and _transcript_is_cached(transcript, metadata, settings)
    )

    if not transcript_cached:
        source = config.resolve(str(metadata.get("source_path") or ""))
        if not source.is_file():
            raise FileNotFoundError(
                f"source video not found for {video_id!r}: {source}. "
                "Keep the source video on disk while indexing speech."
            )
        with StageTimer("audio", _LOG):
            audio_file = extract_audio(source, audio_path(config, video_id), overwrite=True)
        if audio_file is None:
            return (
                _write_empty_sidecar(
                    video_id, config, metadata, settings, reason="no_audio_stream"
                ),
                False,
            )
        with StageTimer("transcribe", _LOG):
            transcript = transcribe_audio(
                audio_file,
                model_name=whisper_model,
                device=device or config.get("speech.device", "auto"),
                compute_type=config.get("speech.compute_type"),
                language=config.get("speech.language"),
                beam_size=int(config.get("speech.beam_size", 5)),
                word_timestamps=bool(config.get("speech.word_timestamps", True)),
                vad_filter=bool(config.get("speech.vad_filter", True)),
                show_progress=show_progress,
            )
        assert transcript is not None
        transcript.update(
            {
                "schema_version": SCHEMA_VERSION,
                "tool": "videoseek",
                "video_id": video_id,
                "source_path": metadata.get("source_path"),
                "source_hash": metadata.get("source_hash"),
                "audio_path": store.relative_path(
                    audio_path(config, video_id), config.repo_root
                ),
                "settings": _transcript_settings(settings),
                "created_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        store.write_json(transcript_file, transcript)
    else:
        assert transcript is not None
        _LOG.info("cache hit: transcript %s", transcript_file)

    chunks = chunk_transcript(
        transcript.get("segments") or [],
        chunk_seconds=settings["chunk_seconds"],
        overlap_seconds=settings["overlap_seconds"],
    )
    if not chunks:
        return (
            _write_empty_sidecar(
                video_id, config, metadata, settings, reason="no_speech_detected"
            ),
            False,
        )

    text_device = resolve_device(config.get("text.device", "auto"))
    encoder = load_text_encoder(embedding_model, text_device)
    with StageTimer("embed chunks", _LOG):
        vectors = encode_texts(
            [chunk["text"] for chunk in chunks],
            encoder,
            batch_size=batch_size,
        )
    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"chunk embedding count {vectors.shape[0]} does not match chunks {len(chunks)}"
        )

    with StageTimer("faiss index", _LOG):
        index = store.build_flat_ip(vectors)
        store.save_faiss(index, index_path)

    records = []
    for position, chunk in enumerate(chunks):
        records.append(
            {
                "doc_id": f"{video_id}:speech:{int(chunk['index']):04d}",
                "video_id": video_id,
                "layer": "speech",
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "path": None,
                "text": chunk["text"],
                "meta": {
                    "chunk_idx": int(chunk["index"]),
                    "word_count": len(chunk["words"]),
                    "words": chunk["words"],
                    "position": position,
                },
            }
        )

    segments = transcript.get("segments") or []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "layer": "speech",
        "model": whisper_model,
        "embedding_model": embedding_model,
        "device": transcript.get("device") or text_device,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "count": len(records),
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "audio_path": transcript.get("audio_path")
        or store.relative_path(audio_path(config, video_id), config.repo_root),
        "transcript_path": store.relative_path(transcript_file, config.repo_root),
        "language": transcript.get("language"),
        "chunk_seconds": settings["chunk_seconds"],
        "overlap_seconds": settings["overlap_seconds"],
        "settings": settings,
        "stats": {
            "segments": len(segments),
            "words": sum(len(segment.get("words") or []) for segment in segments),
            "chunks": len(chunks),
            "duration_s": transcript.get("duration_s"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": records,
    }
    store.write_json(ids_path, payload)
    _LOG.info(
        "wrote %d speech chunks (%d-d) from %d transcript segments to %s",
        len(records),
        payload["dim"],
        len(segments),
        store.relative_path(index_path, config.repo_root),
    )
    return payload, False
