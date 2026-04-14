"""
Tests for the transcript fetcher module.
"""

import pytest
from src.transcript.fetcher import extract_video_id, TranscriptFetchError


class TestExtractVideoId:
    """Test video ID extraction from various URL formats."""

    def test_bare_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_standard_url(self):
        assert extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id(
            "https://youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert extract_video_id(
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_url_with_params(self):
        assert extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        ) == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        assert extract_video_id(
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_whitespace_stripped(self):
        assert extract_video_id("  dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(TranscriptFetchError):
            extract_video_id("https://notavalidsite.com/blah")

    def test_empty_raises(self):
        with pytest.raises(TranscriptFetchError):
            extract_video_id("")
