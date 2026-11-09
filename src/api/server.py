"""FastAPI server for VideoSeek.

Run the API alone with:
    uvicorn src.api.server:app --port 8000

Run the UI + API with:
    python app.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.jobs import get_config, get_job, list_jobs, start_job
from src.index.video import find_ffmpeg, video_id_for
from src.search import service
from src.utils.logging import get_logger

_LOG = get_logger("api.server")

app = FastAPI(
    title="VideoSeek",
    version="1.0.0",
    description="Natural-language search over long video with timestamped segments.",
)

_MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class IndexRequest(BaseModel):
    video_path: str
    layers: Optional[list[str]] = None
    overwrite: bool = False
    video_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    video_id: Optional[str] = None
    layers: str = "auto"
    top_k: int = Field(10, ge=1, le=50)
    adaptive: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    config = get_config()
    videos = service.list_videos(config)
    return {
        "status": "ok",
        "videos": len(videos),
        "layers": ["frames", "objects", "speech", "ocr", "caption"],
    }


@app.get("/videos")
def videos() -> dict[str, Any]:
    config = get_config()
    items = service.list_videos(config)
    return {"count": len(items), "videos": items}


@app.get("/search")
def search_get(
    q: str = Query(..., min_length=1),
    video_id: Optional[str] = None,
    layers: str = "auto",
    top_k: int = Query(10, ge=1, le=50),
    adaptive: bool = True,
) -> dict[str, Any]:
    return _search(q, video_id, layers, top_k, adaptive)


@app.post("/search")
def search_post(request: SearchRequest) -> dict[str, Any]:
    return _search(
        request.query,
        request.video_id,
        request.layers,
        request.top_k,
        request.adaptive,
    )


def _search(
    query: str,
    video_id: Optional[str],
    layers: str,
    top_k: int,
    adaptive: bool,
) -> dict[str, Any]:
    config = get_config()
    if video_id:
        video_id = video_id_for(video_id) if Path(video_id).suffix else video_id
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", video_id):
            raise HTTPException(status_code=400, detail="invalid video_id")
    try:
        return service.search(
            query,
            config,
            video_id=video_id,
            layers=layers,
            top_k=top_k,
            adaptive=adaptive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/index")
def index_video(request: IndexRequest) -> dict[str, Any]:
    try:
        job = start_job(
            request.video_path,
            request.layers,
            overwrite=request.overwrite,
            video_id=request.video_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.as_dict()


@app.get("/index")
def index_jobs() -> dict[str, Any]:
    return {"jobs": [job.as_dict() for job in list_jobs()]}


@app.get("/index/{job_id}")
def index_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.as_dict()


@app.get("/media/{path:path}")
def media(path: str) -> FileResponse:
    config = get_config()
    root = config.data_dir.resolve()
    candidate = (config.repo_root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="media not found") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in _MEDIA_SUFFIXES:
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(candidate)


@app.get("/clip")
def clip(
    video_id: str,
    start: float = Query(..., ge=0),
    end: float = Query(..., gt=0),
) -> FileResponse:
    config = get_config()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", video_id):
        raise HTTPException(status_code=400, detail="invalid video_id")
    duration = float(end) - float(start)
    if duration <= 0.2:
        raise HTTPException(status_code=400, detail="clip must be longer than 0.2 s")
    duration = min(duration, 30.0)
    end = float(start) + duration

    source = service.resolve_source(config, video_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source video not available")

    clips_dir = config.data_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_path = clips_dir / f"{video_id}_{start:.2f}_{end:.2f}.mp4"
    if not out_path.is_file():
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise HTTPException(status_code=500, detail="ffmpeg not found")
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0 or not out_path.is_file():
            out_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"clip extraction failed: {(result.stderr or '')[:300]}",
            )
    return FileResponse(out_path, media_type="video/mp4", filename=out_path.name)
