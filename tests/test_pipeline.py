"""
Integration tests for the full pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.transcript.preprocessor import (
    remove_fillers,
    restore_punctuation,
    merge_segments,
    preprocess_transcript,
)


class TestPreprocessor:
    """Test transcript preprocessing."""

    def test_remove_fillers(self):
        text = "um so basically uh we need to like you know process the data"
        cleaned = remove_fillers(text)
        assert "um" not in cleaned.split()
        assert "uh" not in cleaned.split()

    def test_restore_punctuation_capitalize(self):
        text = "hello world"
        result = restore_punctuation(text)
        assert result[0] == "H"
        assert result.endswith(".")

    def test_restore_punctuation_preserves_existing(self):
        text = "Hello! How are you?"
        result = restore_punctuation(text)
        assert result == text

    def test_merge_segments_basic(self):
        segments = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
            {"text": "this is", "start": 2.0, "duration": 1.0},
        ]
        merged = merge_segments(segments, max_gap_seconds=2.0)

        # All segments should merge into one since gaps are small
        assert len(merged) == 1
        assert "Hello" in merged[0]["text"]
        assert "world" in merged[0]["text"]

    def test_merge_segments_with_gap(self):
        segments = [
            {"text": "First part.", "start": 0.0, "duration": 1.0},
            {"text": "Second part.", "start": 10.0, "duration": 1.0},
        ]
        merged = merge_segments(segments, max_gap_seconds=2.0, min_segment_length=5)

        # Large gap should prevent merging
        assert len(merged) == 2

    def test_preprocess_full_pipeline(self):
        segments = [
            {"text": "um hello welcome", "start": 0.0, "duration": 2.0},
            {"text": "to the lecture", "start": 2.0, "duration": 2.0},
            {"text": "uh today we discuss ml", "start": 4.0, "duration": 3.0},
        ]

        result = preprocess_transcript(segments)
        assert len(result) >= 1
        assert all("text" in seg for seg in result)
        assert all("start" in seg for seg in result)
        assert all("end" in seg for seg in result)

    def test_preprocess_removes_empty(self):
        segments = [
            {"text": "um uh", "start": 0.0, "duration": 1.0},
            {"text": "Hello world", "start": 5.0, "duration": 1.0},
        ]

        result = preprocess_transcript(segments, merge=False)
        # First segment may become empty after filler removal
        assert all(seg["text"].strip() for seg in result)
