"""
End-to-end pipeline orchestrator.

Provides two main workflows:
    1. IngestPipeline: video URL → transcript → clean → chunk → embed → index
    2. QueryPipeline:  question → retrieve → (rerank) → generate → response

Configuration-driven via PipelineConfig dataclass, enabling systematic
variation in ablation studies.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

from config import PipelineConfig, GenerationConfig, LLM_MODELS, TRANSCRIPTS_DIR, INDICES_DIR
from src.transcript.fetcher import fetch_transcript, extract_video_id
from src.transcript.preprocessor import preprocess_transcript
from src.chunking.fixed_chunker import fixed_chunk
from src.chunking.sentence_chunker import sentence_chunk
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.llm import create_llm, LLMConnectionError
from src.generation.chain import RAGChain
from src.generation.hallucination import score_sentences

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Ingests a YouTube video: fetches transcript, preprocesses,
    chunks, embeds, and builds a FAISS index.
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Args:
            config: Pipeline configuration.
            progress_callback: Optional callback(stage_name, progress_fraction).
        """
        self.config = config
        self._progress = progress_callback or (lambda s, p: None)
        self.embedder = Embedder(config.embedding_model)
        self.store = FAISSStore(self.embedder)

    def ingest(self, video_url: str) -> dict:
        """
        Run the full ingestion pipeline for a single video.

        Args:
            video_url: YouTube URL or video ID.

        Returns:
            Dict with summary info:
                - video_id, num_segments, num_chunks, index_path
        """
        video_id = extract_video_id(video_url)

        # Stage 1: Fetch transcript
        self._progress("Fetching transcript", 0.0)
        raw_segments = fetch_transcript(video_id)
        logger.info(f"Fetched {len(raw_segments)} raw segments for {video_id}")

        # Save raw transcript
        raw_path = TRANSCRIPTS_DIR / f"{video_id}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(raw_segments, f, indent=2)

        # Stage 2: Preprocess
        self._progress("Cleaning transcript", 0.2)
        processed_segments = preprocess_transcript(raw_segments)
        logger.info(f"Preprocessed to {len(processed_segments)} segments")

        # Save cleaned transcript
        clean_path = TRANSCRIPTS_DIR / f"{video_id}_clean.json"
        with open(clean_path, "w") as f:
            json.dump(processed_segments, f, indent=2)

        # Stage 3: Chunk
        self._progress("Chunking transcript", 0.4)
        if self.config.chunking.strategy == "sentence":
            chunks = sentence_chunk(
                processed_segments, video_id, self.config.chunking
            )
        else:
            chunks = fixed_chunk(
                processed_segments, video_id, self.config.chunking
            )
        logger.info(
            f"Created {len(chunks)} chunks "
            f"(strategy={self.config.chunking.strategy}, "
            f"size={self.config.chunking.chunk_size})"
        )

        # Stage 4: Embed and build index
        self._progress("Embedding chunks", 0.6)
        self.store.build_index(chunks)
        logger.info(
            f"Built FAISS index with {self.store.num_chunks} vectors "
            f"(dim={self.embedder.dimension})"
        )

        # Stage 5: Save index
        self._progress("Saving index", 0.9)
        index_name = f"{video_id}_{self.config.index_id}"
        index_path = self.store.save(index_name)
        logger.info(f"Saved index to {index_path}")

        self._progress("Complete", 1.0)

        return {
            "video_id": video_id,
            "num_segments": len(processed_segments),
            "num_chunks": len(chunks),
            "index_name": index_name,
            "index_path": str(index_path),
            "embedding_dim": self.embedder.dimension,
        }


class QueryPipeline:
    """
    Handles user queries: retrieves context, optionally reranks,
    and generates answers using the configured LLM.
    """

    def __init__(
        self,
        config: PipelineConfig,
        store: FAISSStore,
        embedder: Embedder,
        skip_llm_health_check: bool = False,
    ):
        """
        Args:
            config: Pipeline configuration.
            store: Pre-built or loaded FAISSStore.
            embedder: Embedder instance (must match the store's model).
            skip_llm_health_check: Skip Ollama connectivity check.
        """
        self.config = config
        self.store = store
        self.embedder = embedder

        # Retriever
        self.retriever = Retriever(store, config.retrieval)

        # Optional reranker
        self.reranker = (
            Reranker(config.retrieval) if config.retrieval.use_reranker
            else None
        )

        # Primary LLM and chain
        self.llm = create_llm(
            config.generation,
            skip_health_check=skip_llm_health_check,
        )
        self.chain = RAGChain(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker,
            config=config,
        )

        # Secondary LLM and chain (the other model from the registry)
        llm_keys = list(LLM_MODELS.keys())
        primary_key = config.generation.model_name
        secondary_key = next((k for k in llm_keys if k != primary_key), primary_key)
        secondary_gen_config = GenerationConfig(
            model_name=secondary_key,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            max_tokens=config.generation.max_tokens,
            ollama_base_url=config.generation.ollama_base_url,
        )
        self.secondary_chain = None
        self.secondary_model_error: str | None = None
        try:
            self.secondary_llm = create_llm(
                secondary_gen_config,
                skip_health_check=skip_llm_health_check,
            )
            self.secondary_chain = RAGChain(
                llm=self.secondary_llm,
                retriever=self.retriever,
                reranker=self.reranker,
                config=config,
            )
        except LLMConnectionError as e:
            self.secondary_model_error = str(e)
        self._primary_model_key = primary_key
        self._secondary_model_key = secondary_key

    def ask(self, question: str, prompt_style: str = "default") -> dict:
        """
        Ask a question about the ingested video.

        Returns dict with: answer, sources, confidence, sentence_scores.
        """
        result = self.chain.ask(question, prompt_style=prompt_style)
        result["sentence_scores"] = score_sentences(
            result["answer"], result["sources"]
        )
        return result

    def ask_dual(self, question: str, prompt_style: str = "default") -> dict:
        """
        Ask a question and get responses from both models.

        Retrieval is performed once and shared. Each model generates
        independently from the same context.

        Returns:
            Dict with keys:
                - question (str)
                - primary (dict): model key, answer, sources
                - secondary (dict): model key, answer, sources
                - candidates (list): retrieved chunks (for committing history later)
        """
        if self.secondary_chain is None:
            raise LLMConnectionError(
                self.secondary_model_error
                or f"Model '{self._secondary_model_key}' is unavailable."
            )

        from src.generation.prompts import format_context

        standalone = self.chain._condense_question(question)

        candidates = self.retriever.retrieve(standalone)
        if self.reranker and self.config.retrieval.use_reranker:
            candidates = self.reranker.rerank(
                standalone, candidates, top_k=self.config.retrieval.top_k
            )
        else:
            candidates = candidates[: self.config.retrieval.top_k]

        context = format_context(candidates)

        primary_result = self.chain.generate_from_context(
            question, context, candidates, prompt_style=prompt_style
        )
        secondary_result = self.secondary_chain.generate_from_context(
            question, context, candidates, prompt_style=prompt_style
        )

        primary_result["sentence_scores"] = score_sentences(
            primary_result["answer"], primary_result["sources"]
        )
        secondary_result["sentence_scores"] = score_sentences(
            secondary_result["answer"], secondary_result["sources"]
        )

        return {
            "question": question,
            "primary": {"model": self._primary_model_key, **primary_result},
            "secondary": {"model": self._secondary_model_key, **secondary_result},
            "candidates": candidates,
        }

    def commit_dual_selection(self, question: str, selected_answer: str):
        """
        After the user picks a response from ask_dual, record it in
        both chains' memory so future follow-ups are contextually aware.
        """
        self.chain.update_memory(question, selected_answer)
        self.secondary_chain.update_memory(question, selected_answer)

    def ask_batch(self, questions: list[str]) -> list[dict]:
        """
        Run multiple questions in batch (for evaluation).
        Resets memory between questions.

        Args:
            questions: List of questions.

        Returns:
            List of result dicts.
        """
        results = []
        for q in questions:
            self.chain.reset_memory()
            results.append(self.ask(q))
        return results

    def retrieve_only(self, question: str) -> list[dict]:
        """
        Retrieve context chunks without generating an answer.
        Used for retrieval evaluation.

        Args:
            question: User's question.

        Returns:
            List of chunk dicts with scores and metadata.
        """
        candidates = self.retriever.retrieve(question)

        if self.reranker and self.config.retrieval.use_reranker:
            candidates = self.reranker.rerank(
                question,
                candidates,
                top_k=self.config.retrieval.top_k,
            )
        else:
            candidates = candidates[:self.config.retrieval.top_k]

        return candidates

    def reset(self):
        """Reset conversation memory."""
        self.chain.reset_memory()


def build_query_pipeline(
    video_id: str,
    config: PipelineConfig,
    skip_llm_health_check: bool = False,
) -> QueryPipeline:
    """
    Convenience function: loads a saved index and creates a QueryPipeline.

    Args:
        video_id: YouTube video ID.
        config: Pipeline configuration.
        skip_llm_health_check: Skip Ollama connectivity check.

    Returns:
        Ready-to-use QueryPipeline.
    """
    embedder = Embedder(config.embedding_model)
    store = FAISSStore(embedder)

    index_name = f"{video_id}_{config.index_id}"
    store.load(index_name)

    return QueryPipeline(
        config=config,
        store=store,
        embedder=embedder,
        skip_llm_health_check=skip_llm_health_check,
    )
