"""
Tests for the embedding module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.embedding.embedder import Embedder


class TestEmbedder:
    """Test the Embedder wrapper."""

    @pytest.fixture
    def embedder(self):
        """Create a MiniLM embedder (smallest, fastest)."""
        return Embedder("minilm")

    def test_initialization(self, embedder):
        assert embedder.model_key == "minilm"
        assert embedder.dimension == 384
        assert embedder.model is not None

    def test_embed_query(self, embedder):
        embedding = embedder.embed_query("What is machine learning?")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_documents(self, embedder):
        texts = ["Hello world", "Machine learning is great", "Python programming"]
        embeddings = embedder.embed_documents(texts, show_progress=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_l2_normalization(self, embedder):
        """Embeddings should be L2-normalized (unit vectors)."""
        embeddings = embedder.embed_documents(
            ["test sentence"], show_progress=False
        )
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_semantic_similarity(self, embedder):
        """Similar texts should have higher cosine similarity."""
        e1 = embedder.embed_query("What is deep learning?")
        e2 = embedder.embed_query("Tell me about neural networks")
        e3 = embedder.embed_query("How to bake a chocolate cake")

        # Cosine sim (already L2-normed, so dot product = cosine)
        sim_related = np.dot(e1, e2)
        sim_unrelated = np.dot(e1, e3)

        assert sim_related > sim_unrelated

    def test_empty_document_list(self, embedder):
        embeddings = embedder.embed_documents([], show_progress=False)
        assert len(embeddings) == 0

    def test_repr(self, embedder):
        r = repr(embedder)
        assert "MiniLM" in r
        assert "384" in r
