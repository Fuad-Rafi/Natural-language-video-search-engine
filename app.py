#!/usr/bin/env python
"""Launch the VideoSeek FastAPI + Gradio app.

    python app.py

Environment:
    VIDEOSEEK_HOST   default 127.0.0.1
    VIDEOSEEK_PORT   default 7860
    VIDEOSEEK_CONFIG optional YAML config path
"""

from __future__ import annotations

import os

import gradio as gr
import uvicorn

from src.api.server import app as api_app
from src.ui.app import build_ui

app = gr.mount_gradio_app(api_app, build_ui(), path="/")


def main() -> None:
    host = os.environ.get("VIDEOSEEK_HOST", "127.0.0.1")
    port = int(os.environ.get("VIDEOSEEK_PORT", "7860"))
    print(f"VideoSeek UI + API: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
