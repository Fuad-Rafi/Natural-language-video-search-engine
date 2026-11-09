"""Unit tests for UI query routing and helper functions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Config, load_config
from src.search import service
from src.search.service import (
    _nearest_frame_path,
    _restrict_to_videos,
    infer_layers,
    is_selectable_source,
    segment_relevance,
    select_adaptive,
    selectable_videos,
)


class RoutingTests(unittest.TestCase):
    def test_speech_wins_over_people(self) -> None:
        self.assertEqual(
            infer_layers("where he says young people who break the law"),
            ["speech"],
        )

    def test_ocr_wins_over_objects(self) -> None:
        self.assertEqual(
            infer_layers("the sign that says red car"),
            ["ocr"],
        )

    def test_object_query(self) -> None:
        self.assertEqual(infer_layers("a red car parked on the street"), ["frame", "object"])

    def test_action_query_adds_caption_and_object(self) -> None:
        layers = infer_layers("a man making espresso")
        self.assertIn("frame", layers)
        self.assertIn("caption", layers)
        self.assertIn("object", layers)

    def test_generic_visual_query_is_frames_only(self) -> None:
        self.assertEqual(infer_layers("an empty road at night"), ["frame"])

    def test_available_layers_drop_missing(self) -> None:
        self.assertEqual(
            infer_layers("where he says hello", available=["frames", "objects"]),
            ["frame"],
        )

    def test_command_does_not_match_man(self) -> None:
        self.assertNotIn("caption", infer_layers("the command sfc /scannow"))


class FrameLookupTests(unittest.TestCase):
    def test_nearest_frame(self) -> None:
        metadata = {
            "frames": [
                {"timestamp_s": 0.0, "path": "a.jpg"},
                {"timestamp_s": 5.0, "path": "b.jpg"},
                {"timestamp_s": 10.0, "path": "c.jpg"},
            ]
        }
        self.assertEqual(_nearest_frame_path(metadata, 6.2), "b.jpg")
        self.assertEqual(_nearest_frame_path(metadata, 9.9), "c.jpg")
        self.assertIsNone(_nearest_frame_path({"frames": []}, 1.0))


class SelectableVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.config = Config(
            {"paths": {"data_dir": "data"}},
            path=self.root / "config.yaml",
            repo_root=self.root,
        )
        self.good = self.root / "data" / "videos" / "good.mp4"
        self.good.parent.mkdir(parents=True)
        self.good.write_bytes(b"video")
        self.benchmark = self.root / "data" / "public" / "charades" / "SLHDI.mp4"
        self.benchmark.parent.mkdir(parents=True)
        self.benchmark.write_bytes(b"video")
        self.missing = self.root / "data" / "videos" / "gone.mp4"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_source_must_exist(self) -> None:
        self.assertTrue(is_selectable_source(str(self.good), self.config))
        self.assertFalse(is_selectable_source(str(self.missing), self.config))
        self.assertFalse(is_selectable_source(None, self.config))

    def test_benchmark_sources_are_hidden(self) -> None:
        self.assertFalse(is_selectable_source(str(self.benchmark), self.config))

    def test_selectable_videos_filters_items(self) -> None:
        videos = [
            {"video_id": "good", "source_path": str(self.good)},
            {"video_id": "gone", "source_path": str(self.missing)},
            {"video_id": "SLHDI", "source_path": str(self.benchmark)},
        ]
        selected = selectable_videos(self.config, videos)
        self.assertEqual([item["video_id"] for item in selected], ["good"])


class SearchScopeTests(unittest.TestCase):
    def test_restrict_hits_to_allowed_videos(self) -> None:
        hits = [
            {"video_id": "visible"},
            {"video_id": "hidden"},
            {"video_id": None},
        ]
        self.assertEqual(
            _restrict_to_videos(hits, {"visible"}),
            [{"video_id": "visible"}],
        )
        self.assertEqual(_restrict_to_videos(hits, None), hits)

    def test_only_selectable_drops_hidden_hits(self) -> None:
        config = load_config()
        layer_hits = [
            {
                "video_id": "visible",
                "layer": "frame",
                "score": 0.12,
                "timestamp": 1.0,
                "start": 1.0,
                "end": 2.0,
                "path": "data/frames/visible/000001.jpg",
            },
            {
                "video_id": "hidden",
                "layer": "frame",
                "score": 0.50,
                "timestamp": 1.0,
                "start": 1.0,
                "end": 2.0,
                "path": "data/frames/hidden/000001.jpg",
            },
        ]
        with patch.object(
            service,
            "selectable_videos",
            return_value=[{"video_id": "visible"}],
        ), patch.object(service, "_search_layer", return_value=layer_hits):
            payload = service.search(
                "test", config, layers="frames", top_k=5, only_selectable=True
            )
        self.assertEqual(payload["videos"], ["visible"])
        self.assertEqual(
            [segment["video_id"] for segment in payload["segments"]],
            ["visible"],
        )


class AdaptiveResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    @staticmethod
    def _segment(evidence: dict, score: float = 0.04) -> dict:
        return {
            "video_id": "vid",
            "start": 0.0,
            "end": 1.0,
            "score": score,
            "layers": [evidence.get("layer", "frame")],
            "evidence": evidence,
        }

    def test_single_strong_match_is_not_padded(self) -> None:
        segments = [
            self._segment({"layer": "frame", "score": 0.12}),
            self._segment({"layer": "frame", "score": 0.06}),
            self._segment({"layer": "frame", "score": 0.05}),
        ]
        selected = select_adaptive(segments, self.config, max_results=5)
        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0].get("weak"))

    def test_multiple_relevant_results_are_kept(self) -> None:
        segments = [
            self._segment({"layer": "frame", "score": 0.125}),
            self._segment({"layer": "frame", "score": 0.12}),
            self._segment({"layer": "frame", "score": 0.11}),
        ]
        selected = select_adaptive(segments, self.config, max_results=5)
        self.assertEqual(len(selected), 3)

    def test_cap_is_respected(self) -> None:
        segments = [
            self._segment({"layer": "frame", "score": 0.125}) for _ in range(6)
        ]
        self.assertEqual(
            len(select_adaptive(segments, self.config, max_results=3)), 3
        )

    def test_weak_query_returns_one_best_guess(self) -> None:
        segments = [
            self._segment({"layer": "frame", "score": 0.04}),
            self._segment({"layer": "frame", "score": 0.03}),
        ]
        selected = select_adaptive(segments, self.config, max_results=5)
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["weak"])

    def test_text_relevance_needs_dense_and_lexical_evidence(self) -> None:
        weak = segment_relevance(
            self._segment(
                {"layer": "speech", "dense_score": 0.46, "bm25_score": 0.0}
            ),
            self.config,
        )
        strong = segment_relevance(
            self._segment(
                {"layer": "speech", "dense_score": 0.75, "bm25_score": 20.0}
            ),
            self.config,
        )
        self.assertLess(weak, 0.30)
        self.assertGreater(strong, 0.66)

    def test_fusion_uses_best_visual_and_text_signals(self) -> None:
        fused = self._segment(
            {
                "layer": "fusion",
                "best_cosine": 0.12,
                "best_dense": 0.44,
                "best_bm25": 0.0,
            }
        )
        self.assertGreater(segment_relevance(fused, self.config), 0.66)

    def test_disabled_returns_exact_top_k(self) -> None:
        segments = [
            self._segment({"layer": "frame", "score": 0.04}) for _ in range(5)
        ]
        selected = select_adaptive(
            segments, self.config, max_results=3, enabled=False
        )
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
