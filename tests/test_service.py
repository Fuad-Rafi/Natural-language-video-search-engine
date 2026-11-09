"""Unit tests for UI query routing and helper functions."""

from __future__ import annotations

import unittest

from src.search.service import _nearest_frame_path, infer_layers


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


if __name__ == "__main__":
    unittest.main()
