"""Smoke tests for the FastAPI surface (no model loading)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.server import app, service


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_media_guard(self) -> None:
        self.assertEqual(self.client.get("/media/README.md").status_code, 404)
        self.assertEqual(
            self.client.get("/media/data/does-not-exist.jpg").status_code, 404
        )

    def test_missing_job(self) -> None:
        self.assertEqual(self.client.get("/index/nope").status_code, 404)

    def test_clip_missing_source(self) -> None:
        response = self.client.get(
            "/clip", params={"video_id": "does-not-exist", "start": 0, "end": 2}
        )
        self.assertEqual(response.status_code, 404)

    def test_search_post_uses_service(self) -> None:
        fake = {
            "query": "test",
            "layers": ["speech"],
            "videos": ["lecture1"],
            "segments": [{"video_id": "lecture1", "start": 1.0, "end": 2.0}],
            "count": 1,
        }
        with patch.object(service, "search", return_value=fake):
            response = self.client.post(
                "/search", json={"query": "test", "layers": "auto", "top_k": 3}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)

    def test_index_starts_job(self) -> None:
        class FakeJob:
            def as_dict(self):
                return {"job_id": "abc", "status": "queued", "video_id": "vid"}

        with patch("src.api.server.start_job", return_value=FakeJob()):
            response = self.client.post(
                "/index", json={"video_path": "data/videos/vid.mp4"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], "abc")


if __name__ == "__main__":
    unittest.main()
