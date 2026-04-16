"""
Unified embedding interface for sentence-transformers models.

Supports the three models compared in the ablation study:
    - all-MiniLM-L6-v2      (384-dim, fast)
    - paraphrase-mpnet-base-v2 (768-dim, balanced)
    - intfloat/e5-small-v2   (384-dim, instruction-tuned)

For e5 models, automatically prepends the required "query: " / "passage: "
prefixes.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDING_MODELS


class Embedder:
    """
    Wraps a sentence-transformers model with convenience methods
    for encoding documents and queries.
    """

    def __init__(self, model_key: str = "minilm"):
        """
        Args:
            model_key: Key in EMBEDDING_MODELS registry
                       ("minilm", "mpnet", "e5").
        """
        self.model_key = model_key
        self.model_name = EMBEDDING_MODELS[model_key]
        self._is_e5 = "e5" in self.model_name.lower()
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of document/passage texts.

        For e5 models, prepends "passage: " to each text.

        Returns:
            numpy array of shape (n, dim).
        """
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine sim
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query text.

        For e5 models, prepends "query: ".

        Returns:
            numpy array of shape (dim,).
        """
        if self._is_e5:
            query = f"query: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)

    def __repr__(self) -> str:
        return f"Embedder(model={self.model_name}, dim={self.dimension})"
