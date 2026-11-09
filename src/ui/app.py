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
MAX_RESULTS = [3, 5]
DEFAULT_MAX_RESULTS = 5

BADGE_CLASS = {
    "VISUAL": "frame",
    "OBJECT": "object",
    "SPEECH": "speech",
    "TEXT": "ocr",
    "CAPTION": "caption",
}

UI_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    font_mono=["ui-monospace", "Cascadia Code", "Consolas", "monospace"],
).set(
    body_background_fill="#f6f7fb",
    body_background_fill_dark="#f6f7fb",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    body_text_color_subdued="#64748b",
    body_text_color_subdued_dark="#64748b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e2e8f0",
    block_border_color_dark="#e2e8f0",
    block_radius="16px",
    block_shadow="0 10px 30px rgba(15, 23, 42, 0.06)",
    block_shadow_dark="0 10px 30px rgba(15, 23, 42, 0.06)",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#e2e8f0",
    input_border_color_dark="#e2e8f0",
    input_border_color_focus="#c7d2fe",
    input_border_color_focus_dark="#c7d2fe",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
)

UI_CSS = """
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 10px 20px 48px !important;
}
footer { display: none !important; }

/* hero */
#vs-hero { padding: 18px 4px 4px; }
#vs-hero .vs-brand {
  font-size: 36px;
  font-weight: 800;
  letter-spacing: -0.035em;
  line-height: 1.05;
  background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
#vs-hero .vs-tagline {
  margin: 7px 0 0;
  color: #64748b;
  font-size: 15px;
}

/* search panel */
#vs-search-panel {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
  margin-top: 14px;
}
#vs-query textarea {
  border: none !important;
  box-shadow: none !important;
  background: #f8fafc !important;
  border-radius: 12px !important;
  font-size: 16px !important;
  padding: 15px 16px !important;
}
#vs-query textarea:focus {
  background: #ffffff !important;
  box-shadow: 0 0 0 3px #e0e7ff !important;
}
#vs-search-btn {
  border-radius: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 8px 20px rgba(79, 70, 229, 0.28) !important;
  min-height: 50px;
}
#vs-filters {
  align-items: flex-end;
  gap: 10px;
  margin-top: 12px;
}
#vs-video label, #vs-layer label, #vs-max label {
  color: #64748b !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
#vs-refresh {
  border-radius: 11px !important;
  color: #475569 !important;
  border-color: #e2e8f0 !important;
  background: #ffffff !important;
  min-height: 42px;
}

/* results */
#vs-results { margin-top: 18px; min-height: 20px; }
#vs-status { color: #64748b; font-size: 13px; margin: 8px 2px 0; }
#vs-status p { margin: 0; }

.vs-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.vs-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.vs-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 36px rgba(15, 23, 42, 0.10);
}
.vs-media {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  background: #0f172a;
}
.vs-body { padding: 12px 14px 14px; }
.vs-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.vs-time {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
  background: #eef2ff;
  border-radius: 8px;
  padding: 3px 9px;
  white-space: nowrap;
}
.vs-match {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  white-space: nowrap;
}
.vs-match.strong { color: #047857; background: #d1fae5; }
.vs-match.related { color: #92400e; background: #fef3c7; }
.vs-match.weak { color: #9f1239; background: #ffe4e6; }
.vs-badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.vs-badge {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.05em;
  padding: 2px 8px;
  border-radius: 999px;
}
.vs-badge.frame { color: #1d4ed8; background: #dbeafe; }
.vs-badge.object { color: #6d28d9; background: #ede9fe; }
.vs-badge.speech { color: #047857; background: #d1fae5; }
.vs-badge.ocr { color: #b45309; background: #fef3c7; }
.vs-badge.caption { color: #be185d; background: #fce7f3; }
.vs-text {
  margin-top: 10px;
  color: #475569;
  font-size: 13px;
  line-height: 1.45;
}
.vs-video { margin-top: 8px; color: #94a3b8; font-size: 11px; }
.vs-empty {
  background: #ffffff;
  border: 1px dashed #cbd5e1;
  border-radius: 16px;
  padding: 38px 24px;
  text-align: center;
  color: #64748b;
}
.vs-empty .vs-icon { font-size: 28px; margin-bottom: 6px; }
.vs-empty b { display: block; color: #334155; font-size: 15px; margin-bottom: 4px; }
.vs-banner {
  margin-bottom: 12px;
  padding: 10px 14px;
  border: 1px solid #fde68a;
  border-radius: 12px;
  background: #fffbeb;
  color: #92400e;
  font-size: 13px;
}

/* examples and api */
#vs-examples { margin-top: 16px; }
#vs-examples table { border-radius: 12px; overflow: hidden; }
#vs-api { margin-top: 12px; }

/* index tab */
#vs-index-panel {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
  margin-top: 14px;
}
#vs-job-status pre {
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 10px;
  padding: 12px;
  overflow-x: auto;
}
#vs-job-status code { background: transparent; color: inherit; }

/* tabs */
button[role="tab"] {
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}
"""

HERO_HTML = """
<div>
  <div class="vs-brand">VideoSeek</div>
  <p class="vs-tagline">
    Search long videos in plain English and jump straight to the moment.
  </p>
</div>
"""

EMPTY_RESULTS = """
<div class="vs-empty">
  <div class="vs-icon">&#128269;</div>
  <b>Search your videos</b>
  <div>Ask for a scene, a spoken phrase, or text on screen.</div>
</div>
"""

API_MARKDOWN = (
    "**API** — `GET /health`, `GET /videos`, `GET /search?q=...`, "
    "`POST /search`, `POST /index`, `GET /index/{job_id}`, "
    "`GET /clip?video_id=...&start=...&end=...`.\n\n"
    "`/search` accepts `adaptive=false` to return the exact `top_k` instead "
    "of the relevance-filtered count."
)


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
        videos = service.selectable_videos(get_config())
    except Exception:  # noqa: BLE001 - UI must stay usable
        return [ALL_VIDEOS]
    return [ALL_VIDEOS] + [item["video_id"] for item in videos]


def _match_label(segment: dict[str, Any]) -> tuple[str, str]:
    if segment.get("weak"):
        return "Best guess", "weak"
    relevance = float(segment.get("relevance") or 0.0)
    if relevance >= 0.66:
        return "Strong", "strong"
    return "Related", "related"


def _badge_html(badges: list[str]) -> str:
    return "".join(
        f'<span class="vs-badge {BADGE_CLASS.get(badge, "frame")}">'
        f"{_escape(badge)}</span>"
        for badge in badges
    )


def render_results(payload: dict[str, Any]) -> str:
    segments = payload.get("segments") or []
    if not segments:
        return (
            '<div class="vs-empty">'
            '<div class="vs-icon">&#128269;</div>'
            "<b>No results</b>"
            "<div>Try another query, another layer, or index a video first.</div>"
            "</div>"
        )

    cards = []
    for segment in segments:
        label, match_class = _match_label(segment)
        thumb = segment.get("thumbnail_url")
        clip = segment.get("clip_url")
        text = segment.get("text")
        if clip:
            poster = f' poster="{_escape(thumb)}"' if thumb else ""
            media = (
                f'<video class="vs-media" controls preload="none"{poster} '
                f'src="{_escape(clip)}"></video>'
            )
        elif thumb:
            media = (
                f'<img class="vs-media" src="{_escape(thumb)}" alt="result" />'
            )
        else:
            media = ""
        text_html = (
            f'<div class="vs-text">&#8220;{_escape(text)}&#8221;</div>'
            if text
            else ""
        )
        cards.append(
            '<div class="vs-card">'
            f"{media}"
            '<div class="vs-body">'
            '<div class="vs-top">'
            f'<span class="vs-time">{_timestamp(segment.get("start"))} '
            f'&ndash; {_timestamp(segment.get("end"))}</span>'
            f'<span class="vs-match {match_class}">{label}</span>'
            "</div>"
            f'<div class="vs-badges">{_badge_html(segment.get("badges", []))}</div>'
            f"{text_html}"
            f'<div class="vs-video">video &middot; '
            f'{_escape(segment.get("video_id"))}</div>'
            "</div></div>"
        )

    banner = ""
    if not payload.get("strong", True):
        banner = (
            '<div class="vs-banner">No strong match &mdash; '
            "showing the closest result.</div>"
        )
    return (
        banner
        + '<div class="vs-grid">'
        + "".join(cards)
        + "</div>"
    )


def run_search(
    query: str,
    video_label: Optional[str],
    layer: str,
    max_results: int,
) -> tuple[str, str]:
    if not query or not query.strip():
        return (
            '<div class="vs-empty"><b>Type a query first.</b></div>',
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
            top_k=int(max_results or DEFAULT_MAX_RESULTS),
            adaptive=bool(config.get("search.adaptive", True)),
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        return (
            '<div class="vs-empty"><b>Search failed.</b> '
            f"<div>{_escape(exc)}</div></div>",
            f"Error: {exc}",
        )

    count = payload.get("count", 0)
    if not count:
        return render_results(payload), "No results."
    if not payload.get("strong", True):
        return (
            render_results(payload),
            "No strong match — showing the closest result.",
        )
    layers = ", ".join(payload.get("layers") or []) or "none"
    noun = "result" if count == 1 else "results"
    return render_results(payload), f"**{count}** relevant {noun} · {layers}"


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
        gr.HTML(HERO_HTML, elem_id="vs-hero")

        with gr.Tab("Search"):
            with gr.Column(elem_id="vs-search-panel"):
                with gr.Row():
                    query = gr.Textbox(
                        label="Query",
                        show_label=False,
                        container=False,
                        placeholder="Ask anything about your videos…",
                        scale=5,
                        elem_id="vs-query",
                    )
                    search_button = gr.Button(
                        "Search",
                        variant="primary",
                        scale=1,
                        elem_id="vs-search-btn",
                    )
                with gr.Row(elem_id="vs-filters"):
                    video = gr.Dropdown(
                        choices=_video_choices(),
                        value=ALL_VIDEOS,
                        label="Video",
                        scale=3,
                        elem_id="vs-video",
                    )
                    layer = gr.Dropdown(
                        choices=LAYERS,
                        value="auto",
                        label="Layer",
                        scale=3,
                        elem_id="vs-layer",
                    )
                    max_results = gr.Radio(
                        choices=MAX_RESULTS,
                        value=DEFAULT_MAX_RESULTS,
                        label="Max results",
                        scale=2,
                        elem_id="vs-max",
                    )
                    refresh_videos = gr.Button(
                        "↻ Refresh",
                        scale=1,
                        elem_id="vs-refresh",
                    )

            results = gr.HTML(EMPTY_RESULTS, elem_id="vs-results")
            search_status = gr.Markdown("", elem_id="vs-status")

            with gr.Column(elem_id="vs-examples"):
                gr.Examples(
                    examples=[
                        [
                            "the command sfc /scannow in command prompt",
                            ALL_VIDEOS,
                            "auto",
                            DEFAULT_MAX_RESULTS,
                        ],
                        [
                            "where he says young people who break the law",
                            ALL_VIDEOS,
                            "auto",
                            DEFAULT_MAX_RESULTS,
                        ],
                        [
                            "a red car parked on the street",
                            ALL_VIDEOS,
                            "auto",
                            DEFAULT_MAX_RESULTS,
                        ],
                        [
                            "a man making espresso",
                            ALL_VIDEOS,
                            "auto",
                            DEFAULT_MAX_RESULTS,
                        ],
                        [
                            "the application was unable to start correctly 0xc0000142",
                            ALL_VIDEOS,
                            "auto",
                            DEFAULT_MAX_RESULTS,
                        ],
                    ],
                    inputs=[query, video, layer, max_results],
                )

            search_button.click(
                run_search,
                inputs=[query, video, layer, max_results],
                outputs=[results, search_status],
            )
            query.submit(
                run_search,
                inputs=[query, video, layer, max_results],
                outputs=[results, search_status],
            )
            refresh_videos.click(
                lambda: gr.update(choices=_video_choices(), value=ALL_VIDEOS),
                outputs=[video],
            )

        with gr.Tab("Index"):
            with gr.Column(elem_id="vs-index-panel"):
                gr.Markdown(
                    "### Index a video\n"
                    "The job runs in the background; use **Refresh status** "
                    "to follow the stage log."
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
                overwrite = gr.Checkbox(
                    label="Rebuild existing indexes", value=False
                )
                with gr.Row():
                    start_button = gr.Button(
                        "Start indexing", variant="primary", elem_id="vs-start"
                    )
                    refresh_button = gr.Button(
                        "Refresh status", elem_id="vs-refresh-job"
                    )
                job_id = gr.Textbox(label="Job ID", interactive=False)
                job_status = gr.Markdown(
                    "No indexing job yet.", elem_id="vs-job-status"
                )
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

        with gr.Accordion("API endpoints", open=False, elem_id="vs-api"):
            gr.Markdown(API_MARKDOWN)

    return demo
