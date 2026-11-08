"""Unit tests for text tokenization, BM25, and intra-layer RRF."""

from __future__ import annotations

import unittest

from src.index.text import bm25_rank, reciprocal_rank_scores, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self) -> None:
        self.assertEqual(
            tokenize("Gradient-descent, then BIG O!"),
            ["gradient", "descent", "then", "big", "o"],
        )

    def test_empty_text(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])  # type: ignore[arg-type]


class Bm25Tests(unittest.TestCase):
    def test_matching_document_ranks_first(self) -> None:
        corpus = [
            "gradient descent updates the weights",
            "the weather is sunny today",
        ]
        ranked = bm25_rank("gradient descent", corpus)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], 0)

    def test_empty_corpus(self) -> None:
        self.assertEqual(bm25_rank("anything", []), [])

    def test_zero_scores_are_dropped(self) -> None:
        corpus = ["gradient descent", "the weather is sunny"]
        ranked = bm25_rank("xylophone", corpus)
        self.assertEqual(ranked, [])

    def test_limit(self) -> None:
        corpus = ["gradient descent", "gradient descent again"]
        self.assertEqual(len(bm25_rank("gradient descent", corpus, limit=1)), 1)


class ReciprocalRankTests(unittest.TestCase):
    def test_positions_present_in_both_lists_score_higher(self) -> None:
        scores = reciprocal_rank_scores([[0, 1], [1, 0]], k=60)
        self.assertAlmostEqual(scores[0], 1 / 61 + 1 / 62, places=9)
        self.assertAlmostEqual(scores[1], 1 / 62 + 1 / 61, places=9)

    def test_single_list(self) -> None:
        scores = reciprocal_rank_scores([[4, 2]], k=60)
        self.assertEqual(sorted(scores), [2, 4])
        self.assertGreater(scores[4], scores[2])

    def test_weights(self) -> None:
        scores = reciprocal_rank_scores([[0], [1]], weights=[1.0, 3.0], k=60)
        self.assertGreater(scores[1], scores[0])

    def test_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_scores([[0]], k=0)

    def test_weight_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_scores([[0], [1]], weights=[1.0])


if __name__ == "__main__":
    unittest.main()
