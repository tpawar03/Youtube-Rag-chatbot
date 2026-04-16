"""
Tests for the retrieval module (FAISS store + retriever).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from config import RetrievalConfig


SAMPLE_CHUNKS = [
    {"text": "Machine learning is a subset of artificial intelligence.", "video_id": "v1", "start_time": 0, "end_time": 10, "chunk_index": 0},
    {"text": "Deep learning uses neural networks with many layers.", "video_id": "v1", "start_time": 10, "end_time": 20, "chunk_index": 1},
    {"text": "Python is a popular programming language for data science.", "video_id": "v1", "start_time": 20, "end_time": 30, "chunk_index": 2},
    {"text": "Chocolate cake requires flour, sugar, cocoa powder, and eggs.", "video_id": "v1", "start_time": 30, "end_time": 40, "chunk_index": 3},
    {"text": "The Eiffel Tower is located in Paris, France.", "video_id": "v1", "start_time": 40, "end_time": 50, "chunk_index": 4},
]


class TestFAISSStore:
    """Test FAISS vector store operations."""

    @pytest.fixture
    def store(self):
        embedder = Embedder("minilm")
        store = FAISSStore(embedder)
        store.build_index(SAMPLE_CHUNKS)
        return store

    def test_build_index(self, store):
        assert store.num_chunks == len(SAMPLE_CHUNKS)

    def test_query_returns_results(self, store):
        results = store.query("What is machine learning?", k=3)
        assert len(results) == 3
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, (float, np.floating))

    def test_query_relevance(self, store):
        results = store.query("What is deep learning?", k=1)
        top_doc, _ = results[0]
        assert "neural networks" in top_doc.page_content.lower() or "deep learning" in top_doc.page_content.lower()

    def test_query_numpy(self, store):
        query_emb = store.embedder.embed_query("machine learning")
        scores, indices = store.query_numpy(query_emb, k=3)
        assert len(scores) == 3
        assert len(indices) == 3
        assert all(isinstance(i, (int, np.integer)) for i in indices)

    def test_get_chunk(self, store):
        chunk = store.get_chunk(0)
        assert chunk["text"] == SAMPLE_CHUNKS[0]["text"]

    def test_save_and_load(self, store, tmp_path):
        store.save("test_index", directory=tmp_path)

        new_store = FAISSStore(Embedder("minilm"))
        new_store.load("test_index", directory=tmp_path)

        assert new_store.num_chunks == store.num_chunks


class TestRetriever:
    """Test the retriever wrapper."""

    @pytest.fixture
    def retriever(self):
        embedder = Embedder("minilm")
        store = FAISSStore(embedder)
        store.build_index(SAMPLE_CHUNKS)
        config = RetrievalConfig(top_k=3, use_reranker=False)
        return Retriever(store, config)

    def test_retrieve(self, retriever):
        results = retriever.retrieve("What is artificial intelligence?")
        assert len(results) <= 3
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "start_time" in r
            assert "end_time" in r
