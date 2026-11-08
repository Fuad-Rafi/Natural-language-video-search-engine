"""Unit tests for OCR keyframes, dedupe, line filtering, and empty caches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import Config
from src.index.ocr import (
    DEFAULT_TEXT_MODEL,
    _extraction_settings,
    dedupe_ocr_keyframes,
    index_ocr_video,
    normalize_ocr_text,
    ocr_index_paths,
    ocr_path,
    ocr_settings,
    read_frame_lines,
    select_keyframes,
    text_similarity,
)


def _config(root: Path) -> Config:
    return Config({"paths": {"data_dir": "data", "logs_dir": "logs"}}, repo_root=root)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((72, 128, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _frames(count: int) -> list[dict[str, object]]:
    return [
        {
            "timestamp_s": float(index),
            "path": f"data/frames/vid/{index:06d}.jpg",
        }
        for index in range(count)
    ]


class KeyframeTests(unittest.TestCase):
    def test_max_interval_keeps_static_frames_on_a_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root).with_overrides(
                {
                    "ocr": {
                        "keyframe_min_interval_s": 1.0,
                        "keyframe_max_interval_s": 5.0,
                        "scene_threshold": 0.1,
                        "blank_std_threshold": 0.0,
                    }
                }
            )
            frames = _frames(13)
            for frame in frames:
                _write_image(config.resolve(str(frame["path"])), 128)
            selected = select_keyframes(frames, config, show_progress=False)
            self.assertEqual(
                [int(item["timestamp_s"]) for item in selected], [0, 5, 10]
            )

    def test_scene_change_is_kept_before_max_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root).with_overrides(
                {
                    "ocr": {
                        "keyframe_min_interval_s": 1.0,
                        "keyframe_max_interval_s": 100.0,
                        "scene_threshold": 0.1,
                        "blank_std_threshold": 0.0,
                    }
                }
            )
            frames = _frames(10)
            for index, frame in enumerate(frames):
                _write_image(
                    config.resolve(str(frame["path"])), 0 if index < 5 else 255
                )
            selected = select_keyframes(frames, config, show_progress=False)
            self.assertEqual(
                [int(item["timestamp_s"]) for item in selected], [0, 5]
            )


class TextHelperTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(
            normalize_ocr_text("  SFC /Scannow! 0xC0000142 "),
            "sfc scannow 0xc0000142",
        )

    def test_similarity(self) -> None:
        self.assertAlmostEqual(
            text_similarity("the slide", "The  Slide!"), 1.0, places=6
        )
        self.assertLess(text_similarity("alpha beta", "wxyz qrst"), 0.8)


class DedupeTests(unittest.TestCase):
    def _keyframe(self, timestamp: float, text: str, confidence: float = 0.9) -> dict:
        return {
            "timestamp_s": timestamp,
            "path": f"frame_{timestamp}.jpg",
            "lines": [
                {"text": text, "confidence": confidence, "box": [0, 0, 10, 10]}
            ],
        }

    def test_identical_text_extends_one_segment(self) -> None:
        segments = dedupe_ocr_keyframes(
            [self._keyframe(0, "our goal"), self._keyframe(5, "our goal")],
            max_gap_s=8.0,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 5.0)
        self.assertEqual(segments[0]["keyframe_count"], 2)

    def test_different_text_starts_a_new_segment(self) -> None:
        segments = dedupe_ocr_keyframes(
            [
                self._keyframe(0, "essay introductions"),
                self._keyframe(5, "the passive voice"),
            ]
        )
        self.assertEqual(len(segments), 2)

    def test_gap_splits_even_when_text_matches(self) -> None:
        segments = dedupe_ocr_keyframes(
            [self._keyframe(0, "same"), self._keyframe(20, "same")],
            max_gap_s=8.0,
        )
        self.assertEqual(len(segments), 2)

    def test_empty_ocr_is_skipped(self) -> None:
        blank = {"timestamp_s": 0.0, "path": "blank.jpg", "lines": []}
        self.assertEqual(dedupe_ocr_keyframes([blank]), [])

    def test_segment_extends_to_the_next_keyframe(self) -> None:
        segments = dedupe_ocr_keyframes(
            [self._keyframe(0, "first slide"), self._keyframe(5, "second slide")],
            max_gap_s=16.0,
        )
        self.assertEqual(segments[0]["end"], 5.0)

    def test_extension_is_capped(self) -> None:
        segments = dedupe_ocr_keyframes(
            [self._keyframe(0, "first slide"), self._keyframe(100, "later slide")],
            max_gap_s=8.0,
        )
        self.assertEqual(segments[0]["end"], 8.0)

    def test_duration_extends_the_last_segment(self) -> None:
        segments = dedupe_ocr_keyframes(
            [self._keyframe(0, "only slide")],
            max_gap_s=8.0,
            duration_s=5.0,
        )
        self.assertEqual(segments[0]["end"], 5.0)

    def test_higher_confidence_representative_wins(self) -> None:
        segments = dedupe_ocr_keyframes(
            [
                self._keyframe(0, "same text", confidence=0.6),
                self._keyframe(4, "same text", confidence=0.99),
            ]
        )
        self.assertEqual(segments[0]["confidence"], 0.99)
        self.assertEqual(segments[0]["path"], "frame_4.jpg")


class LineFilterTests(unittest.TestCase):
    def test_confidence_filter_and_box(self) -> None:
        class Engine:
            def predict(self, _path: str) -> list[dict]:
                return [
                    {
                        "rec_texts": ["keep", "drop"],
                        "rec_scores": [0.9, 0.2],
                        "rec_boxes": [[1, 2, 3, 4], [0, 0, 1, 1]],
                    }
                ]

        lines = read_frame_lines(Engine(), "anything.jpg", min_confidence=0.5)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "keep")
        self.assertEqual(lines[0]["box"], [1.0, 2.0, 3.0, 4.0])

    def test_non_ascii_is_dropped_by_default(self) -> None:
        class Engine:
            def predict(self, _path: str) -> list[dict]:
                return [
                    {
                        "rec_texts": ["hello", "工作", "big O"],
                        "rec_scores": [0.9, 0.9, 0.9],
                        "rec_boxes": [[0, 0, 1, 1]] * 3,
                    }
                ]

        english = read_frame_lines(Engine(), "anything.jpg", min_confidence=0.5)
        self.assertEqual([line["text"] for line in english], ["hello", "big O"])
        everything = read_frame_lines(
            Engine(), "anything.jpg", min_confidence=0.5, allow_non_ascii=True
        )
        self.assertEqual(len(everything), 3)


class PathTests(unittest.TestCase):
    def test_paths_stay_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            self.assertEqual(ocr_path(config, "vid"), root / "data" / "ocr" / "vid.json")
            index_path, ids_path = ocr_index_paths(config, "vid")
            self.assertEqual(index_path.name, "ocr.faiss")
            self.assertEqual(ids_path.name, "ocr.ids.json")


class EmptyCacheTests(unittest.TestCase):
    def test_empty_ocr_json_writes_empty_sidecar_and_is_cached(self) -> None:
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
            settings = ocr_settings(config, DEFAULT_TEXT_MODEL)
            cache = {
                "source_hash": "sha256:abc",
                "settings": _extraction_settings(settings),
                "segments": [],
                "keyframes": [],
            }
            path = ocr_path(config, "vid")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache), encoding="utf-8")

            payload, reused = index_ocr_video("vid", config, show_progress=False)
            self.assertFalse(reused)
            self.assertTrue(payload["empty"])
            self.assertEqual(payload["reason"], "no_text_detected")
            index_path, ids_path = ocr_index_paths(config, "vid")
            self.assertTrue(ids_path.is_file())
            self.assertFalse(index_path.exists())

            cached, reused_again = index_ocr_video("vid", config, show_progress=False)
            self.assertTrue(reused_again)
            self.assertTrue(cached["empty"])


if __name__ == "__main__":
    unittest.main()
