"""Unit tests for the P2 evaluation metrics and query parsing."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_temporal_iou_identical(self) -> None:
        self.assertAlmostEqual(evaluate.temporal_iou(10.0, 12.0, 10.0, 12.0), 1.0, places=6)

    def test_temporal_iou_disjoint(self) -> None:
        self.assertEqual(evaluate.temporal_iou(0.0, 1.0, 5.0, 6.0), 0.0)

    def test_temporal_iou_partial(self) -> None:
        # overlap 1 s, union 3 s
        self.assertAlmostEqual(evaluate.temporal_iou(0.0, 2.0, 1.0, 3.0), 1 / 3, places=6)

    def test_predicted_window(self) -> None:
        self.assertEqual(evaluate.predicted_window(10.0, 1.0), (9.5, 10.5))

    def test_is_hit_tolerance(self) -> None:
        intervals = [(10.0, 12.0)]
        self.assertTrue(evaluate.is_hit(9.6, intervals, 0.5))
        self.assertFalse(evaluate.is_hit(9.4, intervals, 0.5))
        self.assertTrue(evaluate.is_hit(12.5, intervals, 0.5))

    def test_best_iou_uses_best_interval(self) -> None:
        intervals = [(0.0, 2.0), (10.0, 11.0)]
        score = evaluate.best_iou(10.4, intervals, 1.0)
        self.assertGreater(score, 0.5)


class QueryParsingTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        handle.write(text)
        handle.close()
        return Path(handle.name)

    def test_read_and_group_queries(self) -> None:
        path = self._write(
            "query_id,video_id,query,type,start,end,notes\n"
            'q1,vid1,"red car",visual,10,12,\n'
            'q1,vid1,"red car",visual,20,22,second interval\n'
            'q2,vid1,"a man",object,30,31,\n'
        )
        try:
            rows = evaluate.read_queries(path)
            self.assertEqual(len(rows), 3)
            groups = evaluate.group_queries(rows)
            self.assertEqual(len(groups), 2)
            red = next(group for group in groups if group["query_id"] == "q1")
            self.assertEqual(red["intervals"], [(10.0, 12.0), (20.0, 22.0)])
        finally:
            path.unlink(missing_ok=True)

    def test_end_before_start_is_rejected(self) -> None:
        path = self._write(
            "query_id,video_id,query,type,start,end\nq1,vid1,test,visual,10,5\n"
        )
        try:
            with self.assertRaises(ValueError):
                evaluate.read_queries(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_columns_are_rejected(self) -> None:
        path = self._write("query_id,video_id,query\nq1,vid1,test\n")
        try:
            with self.assertRaises(ValueError):
                evaluate.read_queries(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
