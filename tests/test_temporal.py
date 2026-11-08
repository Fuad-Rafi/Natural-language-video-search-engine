"""Unit tests for temporal merging and duplicate measurement."""

from __future__ import annotations

import unittest

from src.search.temporal import duplicate_rate, merge_hits


def hit(timestamp: float, score: float = 0.5, layer: str = "frame", video: str = "v1") -> dict:
    return {
        "video_id": video,
        "timestamp": timestamp,
        "start": timestamp,
        "end": timestamp,
        "score": score,
        "layer": layer,
        "path": f"{video}/{timestamp}.jpg",
        "doc_id": f"{video}:{layer}:{timestamp}",
    }


class MergeTests(unittest.TestCase):
    def test_merges_hits_within_gap(self) -> None:
        segments = merge_hits([hit(0), hit(2), hit(6)], gap_s=3.0, min_segment_s=1.0)
        self.assertEqual(len(segments), 2)
        first = next(segment for segment in segments if segment["start"] == 0)
        self.assertEqual(first["end"], 2)
        self.assertEqual(first["hit_count"], 2)

    def test_single_hit_gets_minimum_duration(self) -> None:
        segments = merge_hits([hit(10)], min_segment_s=1.0)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start"], 10.0)
        self.assertAlmostEqual(segments[0]["end"], 11.0)

    def test_layers_are_unioned(self) -> None:
        segments = merge_hits(
            [hit(10, layer="frame"), hit(10.5, layer="object")],
            gap_s=3.0,
        )
        self.assertEqual(segments[0]["layers"], ["frame", "object"])

    def test_peak_score_and_thumbnail(self) -> None:
        segments = merge_hits([hit(10, score=0.2), hit(11, score=0.9)])
        self.assertEqual(segments[0]["score"], 0.9)
        self.assertIn("11", segments[0]["thumbnail"])

    def test_videos_are_merged_separately(self) -> None:
        segments = merge_hits([hit(10, video="a"), hit(10, video="b")])
        self.assertEqual(len(segments), 2)

    def test_max_segment_splits(self) -> None:
        segments = merge_hits(
            [hit(value) for value in range(10)],
            gap_s=3.0,
            min_segment_s=1.0,
            max_segment_s=3.0,
        )
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment["end"] - segment["start"] <= 4.0 for segment in segments))


class DuplicateTests(unittest.TestCase):
    def test_duplicate_rate_counts_nearby_results(self) -> None:
        self.assertAlmostEqual(duplicate_rate([hit(10), hit(11), hit(20)], gap_s=3.0), 1 / 3)

    def test_no_duplicates_when_far_apart(self) -> None:
        self.assertEqual(duplicate_rate([hit(10), hit(20), hit(30)], gap_s=3.0), 0.0)

    def test_different_videos_are_not_duplicates(self) -> None:
        self.assertEqual(duplicate_rate([hit(10, video="a"), hit(10, video="b")]), 0.0)


if __name__ == "__main__":
    unittest.main()
