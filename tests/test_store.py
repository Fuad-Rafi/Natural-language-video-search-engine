"""Unit tests for FAISS index persistence and search."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.index import store


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # Orthonormal basis: querying a basis vector must return its own row.
        self.vectors = np.eye(4, dtype=np.float32)

    def test_build_and_search_returns_nearest_row(self) -> None:
        index = store.build_flat_ip(self.vectors)
        self.assertEqual(index.ntotal, 4)
        self.assertEqual(index.d, 4)

        scores, positions = store.search(index, self.vectors[2], top_k=2)
        self.assertEqual(int(positions[0]), 2)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
        self.assertGreaterEqual(float(scores[0]), float(scores[1]))

    def test_search_caps_top_k_at_ntotal(self) -> None:
        index = store.build_flat_ip(self.vectors)
        scores, positions = store.search(index, self.vectors[0], top_k=99)
        self.assertEqual(len(scores), 4)
        self.assertEqual(len(positions), 4)

    def test_vectors_are_normalized(self) -> None:
        index = store.build_flat_ip(np.array([[3.0, 4.0]], dtype=np.float32))
        scores, _ = store.search(index, np.array([3.0, 4.0]), top_k=1)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)

    def test_save_load_roundtrip_and_sidecar(self) -> None:
        index = store.build_flat_ip(self.vectors)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            index_path = store.save_faiss(index, base / "frame.faiss")
            self.assertTrue(index_path.is_file())

            loaded = store.load_faiss(index_path)
            self.assertEqual(loaded.ntotal, index.ntotal)
            _, positions = store.search(loaded, self.vectors[1], top_k=1)
            self.assertEqual(int(positions[0]), 1)

            sidecar = base / "frame.ids.json"
            payload = {"count": 4, "records": [{"doc_id": f"d{i}"} for i in range(4)]}
            store.write_json(sidecar, payload)
            self.assertEqual(store.read_json(sidecar), payload)

    def test_dimension_mismatch_is_rejected(self) -> None:
        index = store.build_flat_ip(self.vectors)
        with self.assertRaises(store.StoreError):
            store.search(index, np.zeros(3, dtype=np.float32), top_k=1)

    def test_empty_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            store.build_flat_ip(np.empty((0, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
