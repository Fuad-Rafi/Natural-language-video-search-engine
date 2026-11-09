"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import unittest

from src.search.fusion import reciprocal_rank_fusion


def hit(
    timestamp: float,
    score: float = 0.5,
    layer: str = "frame",
    video: str = "v1",
    **extra: object,
) -> dict:
    record = {
        "video_id": video,
        "timestamp": timestamp,
        "score": score,
        "layer": layer,
        "path": f"{video}/{layer}/{timestamp}.jpg",
        "doc_id": f"{video}:{layer}:{timestamp}",
    }
    record.update(extra)
    return record


class FusionTests(unittest.TestCase):
    def test_same_moment_is_one_cluster(self) -> None:
        fused = reciprocal_rank_fusion(
            {"frame": [hit(10)], "object": [hit(10.5, layer="object")]},
            k=60,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["layers"], ["frame", "object"])
        self.assertAlmostEqual(fused[0]["score"], 2 / 61, places=6)

    def test_different_moments_stay_separate(self) -> None:
        fused = reciprocal_rank_fusion(
            {"frame": [hit(10)], "object": [hit(30, layer="object")]},
            k=60,
            window_s=2.0,
        )
        self.assertEqual(len(fused), 2)

    def test_weights_change_order(self) -> None:
        fused = reciprocal_rank_fusion(
            {"frame": [hit(10)], "object": [hit(30, layer="object")]},
            k=60,
            weights={"frame": 2.0, "object": 0.5},
        )
        self.assertEqual(fused[0]["timestamp"], 10)

    def test_best_rank_per_layer_is_used_once(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "frame": [hit(10), hit(10.5), hit(11)],
                "object": [hit(10.5, layer="object")],
            },
            k=60,
        )
        self.assertEqual(len(fused), 1)
        # frame rank 1 + object rank 1
        self.assertAlmostEqual(fused[0]["score"], 2 / 61, places=6)
        self.assertEqual(fused[0]["hit_count"], 4)

    def test_cluster_carries_best_raw_signals(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "frame": [hit(10, score=0.12)],
                "speech": [
                    hit(
                        10.5,
                        score=0.03,
                        layer="speech",
                        dense_score=0.71,
                        bm25_score=5.5,
                    )
                ],
            },
            k=60,
        )
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["best_cosine"], 0.12)
        self.assertAlmostEqual(fused[0]["best_dense"], 0.71)
        self.assertAlmostEqual(fused[0]["best_bm25"], 5.5)

    def test_cluster_without_text_has_no_text_signals(self) -> None:
        fused = reciprocal_rank_fusion({"frame": [hit(10, score=0.12)]})
        self.assertIsNone(fused[0]["best_dense"])
        self.assertIsNone(fused[0]["best_bm25"])

    def test_empty_rankings(self) -> None:
        self.assertEqual(reciprocal_rank_fusion({"frame": [], "object": []}), [])

    def test_k_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"frame": [hit(10)]}, k=0)


if __name__ == "__main__":
    unittest.main()
