"""Gradio UI for VideoSeek.

Run through ``python app.py`` so the API media/clip routes are mounted on the
same origin as the UI.
"""

from __future__ import annotations

import html
from typing import Any, Optional

import gradio as gr

from src.api.jobs import get_config, get_job, start_job
from src.search import service

ALL_VIDEOS = "All indexed videos"
LAYERS = [
    ("Auto (route by query)", "auto"),
    ("All layers (RRF)", "all"),
    ("Frames", "frames"),
    ("Objects", "objects"),
    ("Speech", "speech"),
    ("On-screen text", "ocr"),
    ("Captions", "caption"),
]
INDEX_LAYERS = ["frames", "objects", "speech", "ocr", "captions"]


def _timestamp(seconds: Any) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    minutes, secs = divmod(value, 60.0)
    if minutes < 60:
        return f"{int(minutes):02d}:{secs:04.1f}"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{secs:04.1f}"


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _video_choices() -> list[str]:
    try:
        videos = service.list_videos(get_config())
    except Exception:  # noqa: BLE001 - UI must stay usable
        return [ALL_VIDEOS]
    return [ALL_VIDEOS] + [item["video_id"] for item in videos]


def render_results(payload: dict[str, Any]) -> str:
    segments = payload.get("segments") or []
    if not segments:
        return (
            "<div style='padding:24px;border:1px dashed #888;border-radius:10px;'>"
            "<b>No results.</b> Try another query, another layer, or index a video first."
            "</div>"
        )

    cards = []
    for segment in segments:
        badges = " ".join(
            f"<span style='display:inline-block;padding:2px 8px;margin-right:4px;"
            f"border-radius:999px;background:#1f2937;color:#e5e7eb;font-size:12px;'>"
            f"{_escape(badge)}</span>"
            for badge in segment.get("badges", [])
        )
        thumb = segment.get("thumbnail_url")
        clip = segment.get("clip_url")
        text = segment.get("text")
        media = ""
        if clip:
            poster = f' poster="{_escape(thumb)}"' if thumb else ""
            media = (
                f'<video controls preload="none"{poster} '
                f'style="width:100%;border-radius:8px;background:#000;" '
                f'src="{_escape(clip)}"></video>'
            )
        elif thumb:
            media = (
                f'<img src="{_escape(thumb)}" '
                f'style="width:100%;border-radius:8px;display:block;" />'
            )
        text_html = (
            f"<div style='margin-top:8px;color:#d1d5db;font-size:14px;'>"
            f"&#8220;{_escape(text)}&#8221;</div>"
            if text
            else ""
        )
        cards.append(
            f"""
            <div style="border:1px solid #374151;border-radius:12px;padding:12px;
                        background:#111827;overflow:hidden;">
              {media}
              <div style="margin-top:10px;font-size:13px;color:#9ca3af;">
                <b style="color:#f9fafb;">{_timestamp(segment.get('start'))}
                &ndash; {_timestamp(segment.get('end'))}</b>
                &nbsp; {badges}
                <span style="float:right;">score {float(segment.get('score', 0.0)):.3f}</span>
              </div>
              <div style="margin-top:4px;font-size:12px;color:#6b7280;">
                video: {_escape(segment.get('video_id'))}
              </div>
              {text_html}
            </div>
            """
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));"
        "gap:14px;'>" + "".join(cards) + "</div>"
    )


def run_search(
    query: str,
    video_label: Optional[str],
    layer: str,
    top_k: int,
) -> tuple[str, str]:
    if not query or not query.strip():
        return (
            "<div style='padding:16px;'>Type a query first.</div>",
            "No query.",
        )
    config = get_config()
    video_id = None if not video_label or video_label == ALL_VIDEOS else video_label
    try:
        payload = service.search(
            query,
            config,
            video_id=video_id,
            layers=layer or "auto",
            top_k=int(top_k),
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        return (
            f"<div style='padding:16px;color:#fca5a5;'>Search failed: {_escape(exc)}</div>",
            f"Error: {exc}",
        )
    status = (
        f"**{payload['count']}** segments · layers: "
        f"{', '.join(payload['layers']) or 'none'} · videos: "
        f"{', '.join(payload['videos']) or 'none'}"
    )
    return render_results(payload), status


def start_index(
    video_path: str,
    layers: list[str],
    overwrite: bool,
) -> tuple[str, str]:
    if not video_path or not video_path.strip():
        return "", "Enter a video path first."
    try:
        job = start_job(
            video_path.strip(),
            layers or INDEX_LAYERS,
            overwrite=bool(overwrite),
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        return "", f"Could not start indexing: {exc}"
    return job.id, format_job(job.as_dict())


def refresh_index(job_id: str) -> str:
    if not job_id or not job_id.strip():
        return "No job selected."
    job = get_job(job_id.strip())
    if job is None:
        return "Job not found."
    return format_job(job.as_dict())


def format_job(job: dict[str, Any]) -> str:
    lines = [
        f"**Status:** {job.get('status')}",
        f"**Stage:** {job.get('stage')}",
        f"**Video:** {job.get('video_id')}",
        f"**Layers:** {', '.join(job.get('layers') or [])}",
    ]
    if job.get("error"):
        lines.append(f"**Error:** `{job['error']}`")
    if job.get("log"):
        lines.append("```\n" + "\n".join(job["log"][-12:]) + "\n```")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VideoSeek") as demo:
        gr.Markdown(
            "# VideoSeek\n"
            "Search long videos in plain English. Results are timestamped "
            "segments with source badges: **VISUAL**, **OBJECT**, **SPEECH**, "
            "**TEXT**, **CAPTION**. Click a clip to play it."
        )

        with gr.Tab("Search"):
            with gr.Row():
                query = gr.Textbox(
                    label="Query",
                    placeholder="where he says gradient descent",
                    scale=4,
                )
                video = gr.Dropdown(
                    choices=_video_choices(),
                    value=ALL_VIDEOS,
                    label="Video",
                    scale=2,
                )
            with gr.Row():
                layer = gr.Dropdown(
                    choices=LAYERS,
                    value="auto",
                    label="Layer",
                    scale=2,
                )
                top_k = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1, label="Results"
                )
                search_button = gr.Button("Search", variant="primary", scale=1)
                refresh_videos = gr.Button("Refresh videos", scale=1)
            results = gr.HTML(
                "<div style='padding:16px;color:#9ca3af;'>Results appear here.</div>"
            )
            search_status = gr.Markdown("")

            search_button.click(
                run_search,
                inputs=[query, video, layer, top_k],
                outputs=[results, search_status],
            )
            query.submit(
                run_search,
                inputs=[query, video, layer, top_k],
                outputs=[results, search_status],
            )
            refresh_videos.click(
                lambda: gr.update(choices=_video_choices(), value=ALL_VIDEOS),
                outputs=[video],
            )
            gr.Examples(
                examples=[
                    ["the command sfc /scannow in command prompt", ALL_VIDEOS, "auto", 10],
                    ["where he says young people who break the law", ALL_VIDEOS, "auto", 10],
                    ["a red car parked on the street", ALL_VIDEOS, "auto", 10],
                    ["a man making espresso", ALL_VIDEOS, "auto", 10],
                    ["the application was unable to start correctly 0xc0000142", ALL_VIDEOS, "auto", 10],
                ],
                inputs=[query, video, layer, top_k],
            )

        with gr.Tab("Index"):
            gr.Markdown(
                "Index a local video. The job runs in the background; use "
                "**Refresh status** to follow the stage."
            )
            index_path = gr.Textbox(
                label="Video path",
                placeholder="data/videos/lecture1.mp4",
            )
            index_layers = gr.CheckboxGroup(
                choices=INDEX_LAYERS,
                value=INDEX_LAYERS,
                label="Layers",
            )
            overwrite = gr.Checkbox(label="Rebuild existing indexes", value=False)
            with gr.Row():
                start_button = gr.Button("Start indexing", variant="primary")
                refresh_button = gr.Button("Refresh status")
            job_id = gr.Textbox(label="Job ID", interactive=False)
            job_status = gr.Markdown("No indexing job yet.")
            start_button.click(
                start_index,
                inputs=[index_path, index_layers, overwrite],
                outputs=[job_id, job_status],
            )
            refresh_button.click(
                refresh_index,
                inputs=[job_id],
                outputs=[job_status],
            )

        gr.Markdown(
            "API: `GET /health`, `GET /videos`, `GET /search?q=...`, "
            "`POST /search`, `POST /index`, `GET /index/{job_id}`, "
            "`GET /clip?video_id=...&start=...&end=...`."
        )
    return demo
