"""Unit tests for object crop geometry and selection."""

from __future__ import annotations

import unittest

from src.index.objects import (
    _prepare_crop_specs,
    _safe_track_name,
    anchor_tracks,
    classify_crop_color,
    expand_box,
    select_crop_indices,
    split_tracks,
)
from src.search.query import (
    collapse_object_pairs,
    filter_object_matches,
    parse_object_query,
)


class _Config:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)


class GeometryTests(unittest.TestCase):
    def test_expand_box_grows_and_clamps(self) -> None:
        left, top, right, bottom = expand_box([10, 20, 30, 40], 100, 100, 0.5)
        self.assertEqual((left, top, right, bottom), (0, 10, 40, 50))

    def test_expand_box_clamps_at_image_edges(self) -> None:
        left, top, right, bottom = expand_box([90, 90, 100, 100], 100, 100, 0.5)
        self.assertEqual((left, top, right, bottom), (85, 85, 100, 100))

    def test_invalid_box_returns_zero(self) -> None:
        self.assertEqual(expand_box([0, 0, 0, 0], 100, 100, 0.1), (0, 0, 0, 0))

    def test_safe_track_name(self) -> None:
        self.assertEqual(_safe_track_name(-12), "tm00012")
        self.assertEqual(_safe_track_name(7), "t000007")


class CropSelectionTests(unittest.TestCase):
    def _detections(self, count: int) -> list[dict[str, float]]:
        return [{"confidence": 0.1 * (index + 1)} for index in range(count)]

    def test_short_track_keeps_every_crop(self) -> None:
        self.assertEqual(select_crop_indices(self._detections(2), 3), [0, 1])

    def test_long_track_keeps_first_middle_best(self) -> None:
        detections = self._detections(10)
        detections[7]["confidence"] = 2.0
        selected = select_crop_indices(detections, 3)
        self.assertEqual(selected, [0, 5, 7])

    def test_missing_middle_does_not_duplicate(self) -> None:
        detections = self._detections(5)
        detections[2]["confidence"] = 0.99  # best == middle
        selected = select_crop_indices(detections, 3)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertIn(0, selected)
        self.assertIn(2, selected)

    def test_prepare_specs_respects_global_cap(self) -> None:
        tracks = [
            {
                "track_id": index,
                "detections": self._detections(3),
                "start": float(index),
                "end": float(index),
            }
            for index in range(10)
        ]
        config = _Config(
            {
                "objects.max_crops_per_track": 3,
                "objects.max_crops_per_video": 9,
            }
        )
        specs = _prepare_crop_specs(tracks, config)
        self.assertEqual(len(specs), 9)


class TrackSplitTests(unittest.TestCase):
    def _track(self, times: list[float]) -> dict[str, object]:
        detections = [
            {
                "timestamp_s": value,
                "class_id": 0,
                "confidence": 0.9,
                "box": [0, 0, 10, 10],
                "frame_path": "x.jpg",
            }
            for value in times
        ]
        return {
            "track_id": 7,
            "class_id": 0,
            "class_name": "person",
            "start": times[0],
            "end": times[-1],
            "frame_count": len(times),
            "detections": detections,
        }

    def test_short_track_is_one_segment(self) -> None:
        segments = split_tracks([self._track([0, 1, 2, 3])])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[0]["end"], 3)

    def test_gap_splits_segments(self) -> None:
        segments = split_tracks([self._track([0, 1, 2, 10, 11])], max_gap_s=2.5)
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment["start"] for segment in segments], [0, 10])
        self.assertEqual([segment["track_id"] for segment in segments], [1, 2])

    def test_long_span_splits_segments(self) -> None:
        segments = split_tracks(
            [self._track(list(range(0, 30)))], max_gap_s=2.5, max_segment_s=12.0
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["end"], 12)
        self.assertEqual(segments[1]["start"], 13)
        self.assertEqual(segments[-1]["end"], 29)

    def test_segment_can_override_class_name(self) -> None:
        track = self._track([0, 1])
        track["detections"][0]["class_id"] = 2
        track["detections"][1]["class_id"] = 2
        segments = split_tracks([track], class_names={2: "car"})
        self.assertEqual(segments[0]["class_id"], 2)
        self.assertEqual(segments[0]["class_name"], "car")


class AnchorTests(unittest.TestCase):
    def _frame(self, timestamp: float, sizes: list[tuple[float, float]]) -> dict[str, object]:
        return {
            "timestamp_s": timestamp,
            "detections": [
                {
                    "timestamp_s": timestamp,
                    "box": [0, 0, width, height],
                    "confidence": 0.9,
                    "class_id": 0,
                    "frame_path": "x.jpg",
                }
                for width, height in sizes
            ],
        }

    def test_anchor_grid_and_top_k(self) -> None:
        frames = [
            self._frame(0, [(10, 10), (50, 50), (100, 50)]),
            self._frame(1, [(10, 10)]),
            self._frame(6, [(20, 20), (30, 30)]),
        ]
        anchors = anchor_tracks(frames, {0: "person"}, interval_s=6, max_per_frame=2)
        self.assertEqual(len(anchors), 4)
        self.assertEqual([anchor["start"] for anchor in anchors], [0, 0, 6, 6])
        self.assertEqual(anchors[0]["class_name"], "person")
        self.assertTrue(all(anchor["anchor"] for anchor in anchors))
        self.assertLess(anchors[0]["track_id"], 0)

    def test_anchor_disabled(self) -> None:
        frames = [self._frame(0, [(10, 10)])]
        self.assertEqual(anchor_tracks(frames, {0: "person"}, interval_s=0), [])


class QueryConstraintTests(unittest.TestCase):
    def test_parse_color_class_night(self) -> None:
        colors, classes, night = parse_object_query("a white van driving at night")
        self.assertEqual(colors, {"white"})
        self.assertEqual(classes, {"car", "truck"})
        self.assertTrue(night)

    def test_parse_cyclist(self) -> None:
        colors, classes, night = parse_object_query("a cyclist in a green jacket")
        self.assertEqual(colors, {"green"})
        self.assertIn("person", classes or set())
        self.assertFalse(night)

    def test_filter_object_matches_applies_constraints(self) -> None:
        records = [
            {"meta": {"class_name": "car", "color": "red", "brightness": 90}},
            {"meta": {"class_name": "car", "color": "white", "brightness": 90}},
            {"meta": {"class_name": "person", "color": "red", "brightness": 90}},
        ]
        scores = [0.1, 0.2, 0.3]
        positions = [0, 1, 2]
        pairs = filter_object_matches(records, scores, positions, "a red car")
        self.assertEqual([record["meta"]["color"] for _, record in pairs], ["red"])

    def test_filter_falls_back_when_color_missing(self) -> None:
        records = [{"meta": {"class_name": "car", "color": "grey", "brightness": 90}}]
        pairs = filter_object_matches(records, [0.3], [0], "a green car")
        self.assertEqual(len(pairs), 1)


class CollapseTests(unittest.TestCase):
    def _record(self, timestamp: float, box: list[float], name: str = "car") -> dict:
        return {
            "meta": {
                "class_name": name,
                "timestamp_s": timestamp,
                "crop_bbox": box,
            }
        }

    def test_same_object_within_two_seconds_is_collapsed(self) -> None:
        pairs = [
            (0.5, self._record(10, [0, 0, 100, 100])),
            (0.7, self._record(11, [5, 0, 100, 100])),
        ]
        kept = collapse_object_pairs(pairs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], 0.7)

    def test_different_classes_are_kept(self) -> None:
        pairs = [
            (0.5, self._record(10, [0, 0, 100, 100], "car")),
            (0.6, self._record(10, [0, 0, 100, 100], "person")),
        ]
        self.assertEqual(len(collapse_object_pairs(pairs)), 2)

    def test_non_overlapping_boxes_are_kept(self) -> None:
        pairs = [
            (0.5, self._record(10, [0, 0, 50, 50])),
            (0.6, self._record(11, [500, 500, 600, 600])),
        ]
        self.assertEqual(len(collapse_object_pairs(pairs)), 2)

    def test_objects_far_apart_in_time_are_kept(self) -> None:
        pairs = [
            (0.5, self._record(10, [0, 0, 100, 100])),
            (0.6, self._record(20, [0, 0, 100, 100])),
        ]
        self.assertEqual(len(collapse_object_pairs(pairs)), 2)


class ColorTests(unittest.TestCase):
    def _solid(self, bgr: tuple[int, int, int]):
        import numpy as np

        return np.full((80, 80, 3), bgr, dtype=np.uint8)

    def test_primary_colors(self) -> None:
        self.assertEqual(classify_crop_color(self._solid((0, 0, 255))), "red")
        self.assertEqual(classify_crop_color(self._solid((255, 0, 0))), "blue")
        self.assertEqual(classify_crop_color(self._solid((0, 255, 0))), "green")

    def test_white_and_black(self) -> None:
        self.assertEqual(classify_crop_color(self._solid((255, 255, 255))), "white")
        self.assertEqual(classify_crop_color(self._solid((0, 0, 0))), "black")


if __name__ == "__main__":
    unittest.main()
