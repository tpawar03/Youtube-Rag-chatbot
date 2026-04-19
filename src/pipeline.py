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
from dataclasses import replace
from pathlib import Path
from typing import Optional, Callable

from config import PipelineConfig, TRANSCRIPTS_DIR, INDICES_DIR
from src.transcript.fetcher import fetch_transcript, extract_video_id
from src.transcript.preprocessor import preprocess_transcript
from src.chunking.fixed_chunker import fixed_chunk
from src.chunking.sentence_chunker import sentence_chunk
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.llm import create_llm
from src.generation.chain import RAGChain
from src.generation.grounding import score_sentences, answer_is_grounded
from src.generation.prompts import parse_confidence, OFF_TOPIC_MESSAGE
from config import LLM_MODELS as _LLM_MODELS_REGISTRY

# All models built into every QueryPipeline for side-by-side comparison.
# The user picks one after seeing both answers to the first question.
DUAL_MODELS: list[str] = list(_LLM_MODELS_REGISTRY.keys())

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

        One RAGChain is built per model in DUAL_MODELS. Health checks run
        for each, so a missing model surfaces early with a clear message
        rather than during the first question. Both chains share the same
        retriever and reranker — retrieval is done once and generation
        runs in both.
        """
        self.config = config
        self.store = store
        self.embedder = embedder

        self.retriever = Retriever(store, config.retrieval)
        self.reranker = (
            Reranker(config.retrieval) if config.retrieval.use_reranker
            else None
        )

        self.chains: dict[str, RAGChain] = {}
        for model_key in DUAL_MODELS:
            gen = replace(config.generation, model_name=model_key)
            llm = create_llm(gen, skip_health_check=skip_llm_health_check)
            chain_config = replace(config, generation=gen)
            self.chains[model_key] = RAGChain(
                llm=llm,
                retriever=self.retriever,
                reranker=self.reranker,
                config=chain_config,
            )

        # The first chain is also used for overview generation, where
        # the choice of model is not user-facing.
        self.chain: RAGChain = self.chains[DUAL_MODELS[0]]

    @property
    def models(self) -> list[str]:
        return list(self.chains.keys())

    def ask_dual(self, question: str, prompt_style: str = "default") -> dict:
        """
        Ask a question against every model on the SAME retrieved context.

        Retrieval runs once (via the first chain's retriever); every model
        then generates from the shared context. Each chain appends to its
        own chat history so follow-ups can later condense correctly against
        whichever model the user picks.

        Returns:
            {
                "sources": list[dict],
                "candidates": list[dict],
                "responses": {
                    <model_key>: {"answer": str, "grounded_sentences": [...]},
                    ...
                },
            }
        """
        from langchain_core.messages import HumanMessage, AIMessage

        context, candidates, sources, off_topic = self.chain._prepare_context(question)

        # Layer 1: retrieval-side off-topic gate on RAW FAISS scores. Returns
        # a non-dual single-answer shape — Streamlit routes it as a plain
        # assistant message; no pick is needed.
        if off_topic:
            for chain in self.chains.values():
                chain._chat_history.append(HumanMessage(content=question))
                chain._chat_history.append(AIMessage(content=OFF_TOPIC_MESSAGE))
            return {
                "off_topic": True,
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
            }

        # Generate with every model on the shared context, applying the
        # post-gen grounding fallback per model: if an LLM ignored the
        # context and answered from outside knowledge, replace its answer
        # with the refusal so the user can't pick an ungrounded response.
        responses: dict[str, dict] = {}
        grounded_count = 0
        for model_key, chain in self.chains.items():
            raw_answer = chain._generate_with_context(question, context, prompt_style)
            answer, confidence = parse_confidence(raw_answer)

            if answer and not answer_is_grounded(answer, candidates):
                answer = OFF_TOPIC_MESSAGE
                confidence = None
            else:
                grounded_count += 1

            chain._chat_history.append(HumanMessage(content=question))
            chain._chat_history.append(AIMessage(content=answer))
            responses[model_key] = {
                "answer": answer,
                "grounded_sentences": score_sentences(answer, candidates),
                "confidence": confidence,
            }

        # If BOTH models produced ungrounded answers, collapse to the same
        # single-message shape as the retrieval gate — nothing useful to
        # pick between.
        if grounded_count == 0:
            return {
                "off_topic": True,
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
            }

        return {
            "off_topic": False,
            "sources": sources,
            "candidates": candidates,
            "responses": responses,
        }

    def ask_with_model(
        self,
        model_key: str,
        question: str,
        prompt_style: str = "default",
    ) -> dict:
        """
        Ask a question using only the specified model's chain. Used after
        the user has picked a winner — subsequent turns route here, so
        only the chosen chain advances its chat history.
        """
        if model_key not in self.chains:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(self.chains)}"
            )
        return self.chains[model_key].ask(question, prompt_style=prompt_style)

    def ask(self, question: str, prompt_style: str = "default") -> dict:
        """Back-compat alias: single-model ask via the default chain."""
        return self.chain.ask(question, prompt_style=prompt_style)

    def ask_batch(self, questions: list[str]) -> list[dict]:
        """
        Run multiple questions in batch (for evaluation).
        Resets memory between questions. Uses the default chain only.
        """
        results = []
        for q in questions:
            self.chain.reset_memory()
            results.append(self.ask(q))
        return results

    def generate_overview(
        self,
        video_id: str,
        force_regenerate: bool = False,
    ) -> dict:
        """
        Produce (and cache) a video overview: summary, key topics,
        suggested questions. Reads the already-saved clean transcript
        for {video_id} from TRANSCRIPTS_DIR.

        Args:
            video_id: YouTube video ID (must already be ingested).
            force_regenerate: Ignore cache and re-run the LLM.

        Returns:
            {"summary": str, "topics": list[str], "suggested_questions": list[str]}
        """
        cache_path = TRANSCRIPTS_DIR / f"{video_id}_overview.json"
        if cache_path.exists() and not force_regenerate:
            with open(cache_path, "r") as f:
                return json.load(f)

        clean_path = TRANSCRIPTS_DIR / f"{video_id}_clean.json"
        with open(clean_path, "r") as f:
            segments = json.load(f)
        transcript_text = " ".join(s.get("text", "") for s in segments)

        overview = self.chain.generate_overview(transcript_text)

        with open(cache_path, "w") as f:
            json.dump(overview, f, indent=2)

        return overview

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
