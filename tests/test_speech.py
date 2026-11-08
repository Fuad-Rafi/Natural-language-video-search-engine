"""Unit tests for speech chunking, cache behavior, and empty-audio handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config
from src.index.speech import (
    audio_path,
    chunk_transcript,
    compute_type_for,
    index_speech_video,
    speech_index_paths,
    transcript_path,
)


def _words(duration_s: float, step_s: float = 0.5) -> list[dict[str, float | str]]:
    words = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + step_s)
        words.append({"w": f"w{len(words)}", "start": start, "end": end})
        start = end
    return words


class ChunkTests(unittest.TestCase):
    def test_short_transcript_is_one_chunk(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "hello world",
                "words": _words(4.0),
            }
        ]
        chunks = chunk_transcript(segments, chunk_seconds=30.0, overlap_seconds=5.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 4.0)
        self.assertEqual(chunks[0]["text"], " ".join(f"w{i}" for i in range(8)))

    def test_long_transcript_splits_with_overlap(self) -> None:
        segments = [
            {"start": 0.0, "end": 60.0, "text": "", "words": _words(60.0)}
        ]
        chunks = chunk_transcript(segments, chunk_seconds=10.0, overlap_seconds=2.5)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertAlmostEqual(chunks[-1]["end"], 60.0, places=3)
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 11.0)
        starts = [chunk["start"] for chunk in chunks]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(starts), len(set(starts)))
        # Consecutive chunks overlap rather than starting at the previous end.
        self.assertLess(chunks[1]["start"], chunks[0]["end"])

    def test_overlap_at_least_chunk_still_progresses(self) -> None:
        segments = [
            {"start": 0.0, "end": 30.0, "text": "", "words": _words(30.0)}
        ]
        chunks = chunk_transcript(segments, chunk_seconds=5.0, overlap_seconds=5.0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len({chunk["start"] for chunk in chunks}), len(chunks))

    def test_falls_back_to_segment_text_without_words(self) -> None:
        segments = [
            {"start": 2.0, "end": 5.0, "text": "no word timings", "words": []}
        ]
        chunks = chunk_transcript(segments, chunk_seconds=30.0, overlap_seconds=5.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "no word timings")
        self.assertEqual(chunks[0]["start"], 2.0)

    def test_empty_transcript(self) -> None:
        self.assertEqual(chunk_transcript([]), [])

    def test_invalid_chunk_seconds(self) -> None:
        with self.assertRaises(ValueError):
            chunk_transcript([], chunk_seconds=0)


class ComputeTypeTests(unittest.TestCase):
    def test_auto_uses_float16_on_cuda(self) -> None:
        self.assertEqual(compute_type_for("cuda", "auto"), "float16")

    def test_auto_uses_int8_on_cpu(self) -> None:
        self.assertEqual(compute_type_for("cpu", "auto"), "int8")

    def test_explicit_preference_wins(self) -> None:
        self.assertEqual(compute_type_for("cuda", "int8"), "int8")


class PathTests(unittest.TestCase):
    def test_paths_stay_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config({"paths": {"data_dir": "data"}}, repo_root=root)
            self.assertEqual(audio_path(config, "vid"), root / "data" / "audio" / "vid.wav")
            self.assertEqual(
                transcript_path(config, "vid"),
                root / "data" / "transcripts" / "vid.json",
            )
            index_path, ids_path = speech_index_paths(config, "vid")
            self.assertEqual(index_path.name, "speech.faiss")
            self.assertEqual(ids_path.name, "speech.ids.json")


class EmptyAudioTests(unittest.TestCase):
    def _config(self, root: Path) -> Config:
        return Config({"paths": {"data_dir": "data", "logs_dir": "logs"}}, repo_root=root)

    def _write_metadata(self, config: Config, video_id: str, has_audio: bool) -> None:
        frames = config.frames_dir / video_id
        frames.mkdir(parents=True, exist_ok=True)
        (frames / "metadata.json").write_text(
            json.dumps(
                {
                    "frames": [{"path": "data/frames/vid/000000.jpg"}],
                    "source_hash": "sha256:abc",
                    "source_path": "data/videos/vid.mp4",
                    "has_audio": has_audio,
                }
            ),
            encoding="utf-8",
        )

    def test_no_audio_writes_empty_sidecar_and_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._write_metadata(config, "vid", has_audio=False)

            payload, reused = index_speech_video("vid", config)
            self.assertFalse(reused)
            self.assertTrue(payload["empty"])
            self.assertEqual(payload["reason"], "no_audio_stream")
            index_path, ids_path = speech_index_paths(config, "vid")
            self.assertTrue(ids_path.is_file())
            self.assertFalse(index_path.exists())

            cached, reused_again = index_speech_video("vid", config)
            self.assertTrue(reused_again)
            self.assertTrue(cached["empty"])


if __name__ == "__main__":
    unittest.main()
