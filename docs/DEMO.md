# Demo

Generate a ~27 s montage from three search results:

```bash
python scripts/make_demo.py
```

Output: `docs/demo.mp4` (gitignored because it contains source footage).

The current montage:

| # | Query | Layer | Result |
|---|---|---|---|
| 1 | “where he says young people who break the law” | SPEECH | segment at 2:35 |
| 2 | “the command sfc /scannow in command prompt” | TEXT | segment at 2:26 |
| 3 | “a man making espresso” | CAPTION + VISUAL + OBJECT | segment at 2:34 |

Each query gets a 2.5 s title card followed by its top clip, normalised to
720p/30fps with AAC audio.

## Longer storyboard (screen recording)

If you record the Gradio UI instead, keep it under 60 s:

1. **0–5 s** — open `http://127.0.0.1:7860`, show the query box.
2. **5–15 s** — run “where he says young people who break the law”; play the top speech clip.
3. **15–25 s** — run “the command sfc /scannow in command prompt”; play the OCR clip.
4. **25–35 s** — run “a man making espresso”; show the caption/visual/object badges.
5. **35–45 s** — switch layer to **All layers** and run a visual query.
6. **45–60 s** — open the Index tab and show the stage log.

The result cards show a thumbnail, time range, source badges, score, matched
text, and an inline `<video>` clip.
