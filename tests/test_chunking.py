"""
Tests for the chunking modules.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config import ChunkingConfig
from src.chunking.fixed_chunker import fixed_chunk
from src.chunking.sentence_chunker import sentence_chunk


# ── Test data ──
SAMPLE_SEGMENTS = [
    {
        "text": "Welcome to this lecture on machine learning. Today we will cover supervised learning.",
        "start": 0.0,
        "end": 10.0,
    },
    {
        "text": "Supervised learning uses labeled data to train models. The most common algorithms include linear regression and decision trees.",
        "start": 10.0,
        "end": 25.0,
    },
    {
        "text": "Linear regression fits a line to the data. Decision trees split the data into branches based on feature values.",
        "start": 25.0,
        "end": 40.0,
    },
    {
        "text": "Now let's discuss neural networks. Neural networks are inspired by the human brain. They consist of layers of interconnected nodes.",
        "start": 40.0,
        "end": 60.0,
    },
    {
        "text": "Each node applies a transformation to its input and passes the result to the next layer. This is called forward propagation.",
        "start": 60.0,
        "end": 80.0,
    },
]


class TestFixedChunker:
    """Test fixed-size token chunking."""

    def test_basic_chunking(self):
        config = ChunkingConfig(strategy="fixed", chunk_size=50)
        chunks = fixed_chunk(SAMPLE_SEGMENTS, "test_video", config)

        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("video_id" in c for c in chunks)
        assert all(c["video_id"] == "test_video" for c in chunks)

    def test_chunk_metadata(self):
        config = ChunkingConfig(strategy="fixed", chunk_size=50)
        chunks = fixed_chunk(SAMPLE_SEGMENTS, "test_video", config)

        for chunk in chunks:
            assert "start_time" in chunk
            assert "end_time" in chunk
            assert "chunk_index" in chunk
            assert chunk["start_time"] >= 0
            assert chunk["end_time"] >= chunk["start_time"]

    def test_sequential_indices(self):
        config = ChunkingConfig(strategy="fixed", chunk_size=50)
        chunks = fixed_chunk(SAMPLE_SEGMENTS, "test_video", config)

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_larger_chunks_fewer_results(self):
        small_config = ChunkingConfig(strategy="fixed", chunk_size=50)
        large_config = ChunkingConfig(strategy="fixed", chunk_size=200)

        small_chunks = fixed_chunk(SAMPLE_SEGMENTS, "test_video", small_config)
        large_chunks = fixed_chunk(SAMPLE_SEGMENTS, "test_video", large_config)

        assert len(small_chunks) >= len(large_chunks)

    def test_empty_segments(self):
        config = ChunkingConfig(strategy="fixed", chunk_size=50)
        chunks = fixed_chunk([], "test_video", config)
        assert chunks == []


class TestSentenceChunker:
    """Test sentence-boundary chunking."""

    def test_basic_chunking(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=50)
        chunks = sentence_chunk(SAMPLE_SEGMENTS, "test_video", config)

        assert len(chunks) > 0
        assert all("text" in c for c in chunks)

    def test_sentence_boundaries(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=50)
        chunks = sentence_chunk(SAMPLE_SEGMENTS, "test_video", config)

        for chunk in chunks:
            # Each chunk should end with sentence-ending punctuation
            text = chunk["text"].strip()
            assert text[-1] in ".!?", f"Chunk doesn't end with punctuation: ...{text[-20:]}"

    def test_metadata_present(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=50)
        chunks = sentence_chunk(SAMPLE_SEGMENTS, "test_video", config)

        for chunk in chunks:
            assert "video_id" in chunk
            assert "start_time" in chunk
            assert "end_time" in chunk
            assert "chunk_index" in chunk

    def test_empty_segments(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=50)
        chunks = sentence_chunk([], "test_video", config)
        assert chunks == []
