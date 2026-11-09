"""Unit tests for caption keyframes, dedupe, and empty caches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config
from src.index.captions import (
    DEFAULT_CAPTION_MODEL,
    _extraction_settings,
    caption_index_paths,
    caption_path,
    caption_settings,
    dedupe_captions,
    index_caption_video,
    select_caption_keyframes,
)


def _config(root: Path) -> Config:
    return Config({"paths": {"data_dir": "data", "logs_dir": "logs"}}, repo_root=root)


def _frames(count: int) -> list[dict[str, object]]:
    return [
        {"timestamp_s": float(index), "path": f"data/frames/vid/{index:06d}.jpg"}
        for index in range(count)
    ]


class KeyframeTests(unittest.TestCase):
    def test_fixed_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary)).with_overrides(
                {"captions": {"keyframe_interval_s": 5.0, "max_keyframes": 300}}
            )
            selected = select_caption_keyframes(_frames(13), config)
            self.assertEqual(
                [int(item["timestamp_s"]) for item in selected], [0, 5, 10]
            )

    def test_max_keyframes_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary)).with_overrides(
                {"captions": {"keyframe_interval_s": 2.0, "max_keyframes": 3}}
            )
            selected = select_caption_keyframes(_frames(31), config)
            self.assertEqual(len(selected), 3)


class DedupeTests(unittest.TestCase):
    def _entry(self, timestamp: float, caption: str) -> dict:
        return {
            "timestamp_s": timestamp,
            "path": f"frame_{timestamp}.jpg",
            "caption": caption,
        }

    def test_identical_captions_merge(self) -> None:
        segments = dedupe_captions(
            [
                self._entry(0, "a man is sitting at a table"),
                self._entry(5, "a man is sitting at a table"),
            ]
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 5.0)
        self.assertEqual(segments[0]["keyframe_count"], 2)

    def test_different_captions_split(self) -> None:
        segments = dedupe_captions(
            [
                self._entry(0, "a man is making coffee"),
                self._entry(5, "a family is playing a board game"),
            ]
        )
        self.assertEqual(len(segments), 2)

    def test_gap_splits_even_when_caption_matches(self) -> None:
        segments = dedupe_captions(
            [self._entry(0, "same scene"), self._entry(30, "same scene")],
            max_gap_s=10.0,
        )
        self.assertEqual(len(segments), 2)

    def test_empty_caption_is_skipped(self) -> None:
        self.assertEqual(dedupe_captions([self._entry(0, "  ")]), [])

    def test_segment_extends_to_next_keyframe(self) -> None:
        segments = dedupe_captions(
            [
                self._entry(0, "first scene"),
                self._entry(5, "second scene"),
            ],
            max_gap_s=10.0,
        )
        self.assertEqual(segments[0]["end"], 5.0)

    def test_duration_extends_last_segment(self) -> None:
        segments = dedupe_captions(
            [self._entry(0, "only scene")],
            max_gap_s=10.0,
            duration_s=6.0,
        )
        self.assertEqual(segments[0]["end"], 6.0)


class PathAndCacheTests(unittest.TestCase):
    def test_paths_stay_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            self.assertEqual(caption_path(config, "vid"), root / "data" / "captions" / "vid.json")
            index_path, ids_path = caption_index_paths(config, "vid")
            self.assertEqual(index_path.name, "caption.faiss")
            self.assertEqual(ids_path.name, "caption.ids.json")

    def test_empty_caption_json_writes_empty_sidecar_and_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            frames = config.frames_dir / "vid"
            frames.mkdir(parents=True, exist_ok=True)
            (frames / "metadata.json").write_text(
                json.dumps(
                    {
                        "frames": [{"path": "data/frames/vid/000000.jpg"}],
                        "source_hash": "sha256:abc",
                        "source_path": "data/videos/vid.mp4",
                    }
                ),
                encoding="utf-8",
            )
            settings = caption_settings(config, "BAAI/bge-small-en-v1.5")
            cache = {
                "source_hash": "sha256:abc",
                "settings": _extraction_settings(settings),
                "captions": [],
                "segments": [],
            }
            path = caption_path(config, "vid")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache), encoding="utf-8")

            payload, reused = index_caption_video(
                "vid", config, show_progress=False
            )
            self.assertFalse(reused)
            self.assertTrue(payload["empty"])
            self.assertEqual(payload["reason"], "no_captions")
            index_path, ids_path = caption_index_paths(config, "vid")
            self.assertTrue(ids_path.is_file())
            self.assertFalse(index_path.exists())

            cached, reused_again = index_caption_video(
                "vid", config, show_progress=False
            )
            self.assertTrue(reused_again)
            self.assertTrue(cached["empty"])

    def test_default_model_name(self) -> None:
        self.assertIn("blip", DEFAULT_CAPTION_MODEL.lower())


if __name__ == "__main__":
    unittest.main()
