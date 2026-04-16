"""
FAISS vector store wrapper.

Builds, saves, loads, and queries a FAISS index.
Uses langchain_community's FAISS integration for seamless
connection to the retrieval chain.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embedding.embedder import Embedder
from config import INDICES_DIR


class FAISSStore:
    """
    Manages a FAISS index for transcript chunks.

    Provides both low-level numpy operations (for evaluation)
    and LangChain-compatible retriever interface (for the RAG chain).
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[dict] = []
        self._langchain_store: Optional[FAISS] = None

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a FAISS index from a list of chunk dicts.

        Args:
            chunks: List of dicts with at least a "text" key.
                    Additional keys are stored as metadata.
        """
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # Embed all chunks
        embeddings = self.embedder.embed_documents(texts)

        # Build FAISS index (Inner Product on L2-normed vectors = cosine sim)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        # Build LangChain FAISS store
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            documents.append(Document(page_content=chunk["text"], metadata=metadata))

        # Create LangChain-compatible embeddings wrapper
        self._langchain_store = FAISS.from_documents(
            documents,
            embedding=_LangChainEmbeddingWrapper(self.embedder),
        )

    def query(
        self, question: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Query the index and return top-K documents with scores.

        Args:
            question: User query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by relevance.
        """
        if self._langchain_store is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        results = self._langchain_store.similarity_search_with_score(
            question, k=k
        )
        return results

    def query_numpy(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Low-level numpy query for evaluation scripts.

        Returns:
            (scores, indices) arrays of shape (k,).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self._index.search(query_embedding, k)
        return scores[0], indices[0]

    def get_chunk(self, index: int) -> dict:
        """Get a chunk dict by its index."""
        return self._chunks[index]

    def save(self, name: str, directory: Optional[Path] = None) -> Path:
        """
        Save the index and chunk metadata to disk.

        Args:
            name: Name for the saved index.
            directory: Directory to save to (defaults to INDICES_DIR).

        Returns:
            Path to the saved index directory.
        """
        save_dir = (directory or INDICES_DIR) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(save_dir / "index.faiss"))

        # Save chunk metadata
        with open(save_dir / "chunks.json", "w") as f:
            json.dump(self._chunks, f, indent=2)

        # Save LangChain store
        if self._langchain_store:
            self._langchain_store.save_local(str(save_dir / "langchain_faiss"))

        return save_dir

    def load(self, name: str, directory: Optional[Path] = None) -> None:
        """
        Load a previously saved index.

        Args:
            name: Name of the saved index.
            directory: Directory to load from (defaults to INDICES_DIR).
        """
        load_dir = (directory or INDICES_DIR) / name

        # Load FAISS index
        self._index = faiss.read_index(str(load_dir / "index.faiss"))

        # Load chunk metadata
        with open(load_dir / "chunks.json", "r") as f:
            self._chunks = json.load(f)

        # Load LangChain store
        lc_path = load_dir / "langchain_faiss"
        if lc_path.exists():
            self._langchain_store = FAISS.load_local(
                str(lc_path),
                embeddings=_LangChainEmbeddingWrapper(self.embedder),
                allow_dangerous_deserialization=True,
            )

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever interface."""
        if self._langchain_store is None:
            raise RuntimeError("Index not built or loaded.")
        return self._langchain_store.as_retriever(**kwargs)


class _LangChainEmbeddingWrapper(Embeddings):
    """
    Adapts our Embedder to the Embeddings interface expected by
    LangChain's FAISS.from_documents().
    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(
            texts, show_progress=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text).tolist()
