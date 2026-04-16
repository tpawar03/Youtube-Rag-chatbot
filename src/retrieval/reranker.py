"""
Cross-encoder re-ranking.

Takes initial FAISS retrieval candidates and re-ranks them using
a cross-encoder model for higher precision. The cross-encoder
processes (query, document) pairs jointly, enabling deep
cross-attention that captures fine-grained relevance.
"""

from sentence_transformers import CrossEncoder

from config import RetrievalConfig, CROSS_ENCODER_MODEL


class Reranker:
    """
    Re-ranks retrieved documents using a cross-encoder model.
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config
        model_name = config.reranker_model or CROSS_ENCODER_MODEL
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidate chunks using the cross-encoder.

        Args:
            query: User's question.
            candidates: List of chunk dicts from initial retrieval.
                        Must have a "text" key.
            top_k: Number of top results to return.
                   Defaults to config.top_k.

        Returns:
            Re-ranked list of chunk dicts (truncated to top_k),
            with an updated "score" field from the cross-encoder.
        """
        if top_k is None:
            top_k = self.config.top_k

        if not candidates:
            return []

        # Build (query, document) pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort descending
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        # Update the primary score field
        for c in reranked:
            c["score"] = c.pop("rerank_score")

        return reranked[:top_k]
