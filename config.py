"""
Central configuration for the YouTube Video RAG Chatbot.

All hyperparameters, model names, and paths are defined here as dataclasses
so they can be passed around the pipeline and varied systematically in ablation studies.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", DATA_DIR / "transcripts"))
INDICES_DIR = Path(os.getenv("INDICES_DIR", DATA_DIR / "indices"))
EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"

# Ensure directories exist
for d in [DATA_DIR, TRANSCRIPTS_DIR, INDICES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Embedding model registry
# ──────────────────────────────────────────────
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "paraphrase-mpnet-base-v2",
    "e5": "intfloat/e5-small-v2",
}

# ──────────────────────────────────────────────
# Cross-encoder model
# ──────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ──────────────────────────────────────────────
# LLM model registry
# ──────────────────────────────────────────────
LLM_MODELS = {
    "mistral": "mistral",
    "llama2": "llama2:7b",
}

# ──────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: Literal["fixed", "sentence"] = "fixed"
    chunk_size: int = 500          # in tokens
    chunk_overlap_pct: float = 0.1  # overlap as fraction of chunk_size

    @property
    def chunk_overlap(self) -> int:
        return int(self.chunk_size * self.chunk_overlap_pct)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval and reranking."""
    top_k: int = 5                  # final number of chunks returned
    faiss_fetch_k: int = 20         # initial FAISS candidates (before reranking)
    use_reranker: bool = False
    reranker_model: str = CROSS_ENCODER_MODEL

    # Off-topic gate. Checked on RAW FAISS L2-distance scores (lower = more
    # similar) BEFORE reranking, so the threshold stays meaningful regardless
    # of whether the cross-encoder is enabled. If the best-matching chunk's
    # distance is >= this threshold, the question is considered unrelated to
    # the video and the pipeline short-circuits with a refusal.
    # Empirical ranges (minilm): on-topic < 1.0, tech-adjacent 1.2-1.3,
    # clearly off-topic 1.8-2.2. Default 1.3 errs on the strict side.
    off_topic_threshold: float = 1.3


@dataclass
class GenerationConfig:
    """
    Sampling hyperparameters for the LLM.

    Chosen for grounded-but-varied RAG answers:
      - temperature 0.4: enough variation to produce distinct phrasings
        across runs without the drift that sets in above ~0.6.
      - top_p 0.9 + top_k 40: complementary nucleus + top-k truncation,
        prunes the low-probability tail that causes hallucination.
      - repeat_penalty 1.15: slightly above Ollama's 1.1 default —
        discourages the LLM from rehashing phrases within a single answer.
      - no seed: repeated calls genuinely differ.
    """
    model_name: str = "mistral"     # key in LLM_MODELS
    temperature: float = 0.4
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.15
    max_tokens: int = 1024
    ollama_base_url: str = OLLAMA_BASE_URL


@dataclass
class PipelineConfig:
    """
    Top-level configuration that bundles all sub-configs.

    This is the single object passed around the pipeline and varied
    in the ablation study.
    """
    embedding_model: str = "minilm"  # key in EMBEDDING_MODELS
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @property
    def embedding_model_name(self) -> str:
        return EMBEDDING_MODELS[self.embedding_model]

    @property
    def llm_model_name(self) -> str:
        return LLM_MODELS[self.generation.model_name]

    @property
    def index_id(self) -> str:
        """Unique identifier for the FAISS index based on config."""
        return (
            f"{self.embedding_model}"
            f"_{self.chunking.strategy}"
            f"_{self.chunking.chunk_size}"
        )

    def __repr__(self) -> str:
        return (
            f"PipelineConfig("
            f"emb={self.embedding_model}, "
            f"chunk={self.chunking.strategy}-{self.chunking.chunk_size}, "
            f"rerank={self.retrieval.use_reranker}, "
            f"llm={self.generation.model_name})"
        )
