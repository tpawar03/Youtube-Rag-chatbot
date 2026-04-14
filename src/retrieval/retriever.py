"""
Top-K retrieval from the FAISS vector store.

Returns documents with relevance scores and timestamp metadata.
"""

from langchain_core.documents import Document

from src.vectorstore.faiss_store import FAISSStore
from config import RetrievalConfig


class Retriever:
    """
    Wraps FAISSStore with configurable top_k and metadata formatting.
    """

    def __init__(self, store: FAISSStore, config: RetrievalConfig):
        self.store = store
        self.config = config

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query: User's question.

        Returns:
            List of dicts with keys:
                - text (str): Chunk content.
                - score (float): Relevance score.
                - start_time (float): Start timestamp.
                - end_time (float): End timestamp.
                - video_id (str): Source video.
                - chunk_index (int): Original chunk index.
        """
        fetch_k = (
            self.config.faiss_fetch_k
            if self.config.use_reranker
            else self.config.top_k
        )

        results = self.store.query(query, k=fetch_k)

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "text": doc.page_content,
                "score": float(score),
                "start_time": doc.metadata.get("start_time", 0.0),
                "end_time": doc.metadata.get("end_time", 0.0),
                "video_id": doc.metadata.get("video_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", -1),
            })

        return retrieved

    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Retrieve as LangChain Document objects (for chain integration).
        """
        fetch_k = (
            self.config.faiss_fetch_k
            if self.config.use_reranker
            else self.config.top_k
        )
        results = self.store.query(query, k=fetch_k)
        return [doc for doc, _ in results]
