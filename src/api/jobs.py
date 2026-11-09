"""Background indexing jobs for the API and UI."""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import Config, load_config
from src.index.captions import index_caption_video
from src.index.frames import index_frame_video
from src.index.objects import index_object_video
from src.index.ocr import index_ocr_video
from src.index.speech import index_speech_video
from src.index.video import extract_video, video_id_for
from src.utils.logging import get_logger

_LOG = get_logger("api.jobs")

DEFAULT_LAYERS = ("frames", "objects", "speech", "ocr", "captions")
ALLOWED_LAYERS = set(DEFAULT_LAYERS)


@dataclass
class IndexJob:
    id: str
    video_path: str
    video_id: str
    layers: list[str]
    overwrite: bool = False
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"
    error: Optional[str] = None
    log: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "video_path": self.video_path,
            "video_id": self.video_id,
            "layers": self.layers,
            "overwrite": self.overwrite,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "log": self.log[-40:],
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_JOBS: dict[str, IndexJob] = {}
_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="videoseek-index")


def get_config() -> Config:
    return load_config(os.environ.get("VIDEOSEEK_CONFIG") or None)


def start_job(
    video_path: str,
    layers: list[str] | None = None,
    *,
    overwrite: bool = False,
    video_id: Optional[str] = None,
) -> IndexJob:
    source = Path(video_path).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    else:
        source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")

    requested = [layer for layer in (layers or DEFAULT_LAYERS) if layer in ALLOWED_LAYERS]
    if "frames" not in requested:
        requested.insert(0, "frames")
    if not requested:
        raise ValueError("no layers selected")

    job = IndexJob(
        id=uuid.uuid4().hex[:12],
        video_path=str(source),
        video_id=video_id or video_id_for(source),
        layers=requested,
        overwrite=overwrite,
    )
    with _LOCK:
        _JOBS[job.id] = job
    _EXECUTOR.submit(_run_job, job)
    return job


def get_job(job_id: str) -> Optional[IndexJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> list[IndexJob]:
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda job: job.created_at, reverse=True)


def _stage(job: IndexJob, stage: str) -> None:
    job.stage = stage
    job.log.append(f"{datetime.now().strftime('%H:%M:%S')} {stage}")
    _LOG.info("[%s] %s", job.id, stage)


def _run_job(job: IndexJob) -> None:
    job.status = "running"
    config = get_config()
    try:
        _stage(job, "frames")
        extract_video(
            job.video_path,
            config,
            video_id=job.video_id,
            overwrite=job.overwrite,
            show_progress=False,
        )
        if "frames" in job.layers:
            _stage(job, "frame embeddings")
            index_frame_video(
                job.video_id,
                config,
                overwrite=job.overwrite,
                show_progress=False,
            )
        if "objects" in job.layers:
            _stage(job, "objects")
            index_object_video(
                job.video_id,
                config,
                overwrite=job.overwrite,
                show_progress=False,
            )
        if "speech" in job.layers:
            _stage(job, "speech")
            index_speech_video(
                job.video_id,
                config,
                overwrite=job.overwrite,
                show_progress=False,
            )
        if "ocr" in job.layers:
            _stage(job, "ocr")
            index_ocr_video(
                job.video_id,
                config,
                overwrite=job.overwrite,
                show_progress=False,
            )
        if "captions" in job.layers:
            _stage(job, "captions")
            index_caption_video(
                job.video_id,
                config,
                overwrite=job.overwrite,
                show_progress=False,
            )
        job.status = "done"
        job.stage = "done"
    except Exception as exc:  # noqa: BLE001 - job boundary
        job.status = "failed"
        job.error = str(exc)
        job.log.append(traceback.format_exc()[-2000:])
        _LOG.exception("index job %s failed", job.id)
    finally:
        job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
