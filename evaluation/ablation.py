"""
Ablation study runner.

Systematically varies:
    - Chunking strategy: fixed-200, fixed-500, fixed-1000, sentence-500
    - Embedding model: MiniLM, MPNet, E5
    - Retrieval method: FAISS-only vs. FAISS + cross-encoder
    - LLM: Mistral-7B vs. Llama-2-7B

Total configurations: 4 × 3 × 2 × 2 = 48

For each config, runs the full retrieval + generation evaluation
pipeline and saves results.
"""

from __future__ import annotations


import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PipelineConfig,
    ChunkingConfig,
    RetrievalConfig,
    GenerationConfig,
    EVAL_DIR,
    RESULTS_DIR,
    EMBEDDING_MODELS,
    LLM_MODELS,
)
from src.pipeline import IngestPipeline, QueryPipeline, build_query_pipeline
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from evaluation.retrieval_eval import (
    evaluate_retrieval,
    aggregate_retrieval_results,
    load_qa_pairs,
)
from evaluation.generation_eval import evaluate_generation
from evaluation.faithfulness_eval import evaluate_faithfulness

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Ablation grid definition
# ──────────────────────────────────────────────
CHUNKING_CONFIGS = [
    ChunkingConfig(strategy="fixed", chunk_size=200),
    ChunkingConfig(strategy="fixed", chunk_size=500),
    ChunkingConfig(strategy="fixed", chunk_size=1000),
    ChunkingConfig(strategy="sentence", chunk_size=500),
]

EMBEDDING_KEYS = list(EMBEDDING_MODELS.keys())  # minilm, mpnet, e5

RETRIEVAL_CONFIGS = [
    RetrievalConfig(use_reranker=False),
    RetrievalConfig(use_reranker=True),
]

LLM_KEYS = list(LLM_MODELS.keys())  # mistral, llama2


def generate_ablation_configs() -> list[PipelineConfig]:
    """
    Generate all configuration combinations for the ablation study.

    Returns:
        List of PipelineConfig objects (48 total).
    """
    configs = []
    for chunk_cfg, emb_key, ret_cfg, llm_key in product(
        CHUNKING_CONFIGS, EMBEDDING_KEYS, RETRIEVAL_CONFIGS, LLM_KEYS
    ):
        config = PipelineConfig(
            embedding_model=emb_key,
            chunking=chunk_cfg,
            retrieval=ret_cfg,
            generation=GenerationConfig(model_name=llm_key),
        )
        configs.append(config)

    return configs


def run_ablation(
    video_ids: list[str],
    qa_pairs_path: Optional[Path] = None,
    configs: Optional[list[PipelineConfig]] = None,
    skip_ingestion: bool = False,
    skip_generation: bool = False,
    output_prefix: str = "ablation",
) -> dict[str, pd.DataFrame]:
    """
    Run the full ablation study.

    Args:
        video_ids: List of YouTube video IDs to evaluate.
        qa_pairs_path: Path to annotated QA pairs JSON.
        configs: List of configs to evaluate (defaults to full grid).
        skip_ingestion: Skip re-ingestion if indices already exist.
        skip_generation: Only run retrieval evaluation.
        output_prefix: Prefix for output CSV files.

    Returns:
        Dict with DataFrames:
            - "retrieval": All retrieval metrics.
            - "generation": All generation metrics (if not skipped).
            - "faithfulness": Faithfulness metrics (if not skipped).
    """
    if configs is None:
        configs = generate_ablation_configs()

    qa_pairs = load_qa_pairs(qa_pairs_path)

    all_retrieval = []
    all_generation = []
    all_faithfulness = []

    total = len(configs) * len(video_ids)
    completed = 0

    for config in configs:
        for video_id in video_ids:
            completed += 1
            logger.info(
                f"[{completed}/{total}] Running: {config} on {video_id}"
            )
            start_time = time.time()

            try:
                # ── Step 1: Ensure index exists ──
                if not skip_ingestion:
                    try:
                        # Check if index already exists
                        embedder = Embedder(config.embedding_model)
                        store = FAISSStore(embedder)
                        index_name = f"{video_id}_{config.index_id}"
                        store.load(index_name)
                        logger.info(f"  Index exists: {index_name}")
                    except FileNotFoundError:
                        logger.info(f"  Ingesting {video_id} with {config.index_id}")
                        pipeline = IngestPipeline(config=config)
                        pipeline.ingest(video_id)

                # ── Step 2: Retrieval evaluation ──
                video_qa = [
                    qa for qa in qa_pairs if qa.get("video_id") == video_id
                ]
                if video_qa:
                    ret_df = evaluate_retrieval(config, video_qa, video_id)
                    if not ret_df.empty:
                        all_retrieval.append(ret_df)

                # ── Step 3: Generation + Faithfulness ──
                if not skip_generation and video_qa:
                    try:
                        qp = build_query_pipeline(
                            video_id, config, skip_llm_health_check=True
                        )

                        questions = [qa["question"] for qa in video_qa]
                        generated = []
                        contexts_list = []

                        for q in questions:
                            qp.reset()
                            chunks = qp.retrieve_only(q)
                            context_texts = [c["text"] for c in chunks]
                            contexts_list.append(context_texts)

                            result = qp.ask(q)
                            generated.append(result["answer"])

                        # Generation metrics
                        gen_df = evaluate_generation(config, video_qa, generated)
                        if not gen_df.empty:
                            all_generation.append(gen_df)

                        # Faithfulness metrics
                        faith_df = evaluate_faithfulness(
                            video_qa, generated, contexts_list, config
                        )
                        if not faith_df.empty:
                            all_faithfulness.append(faith_df)

                    except Exception as e:
                        logger.error(f"  Generation eval failed: {e}")

            except Exception as e:
                logger.error(f"  Failed: {e}")

            elapsed = time.time() - start_time
            logger.info(f"  Completed in {elapsed:.1f}s")

    # ── Combine & save results ──
    results = {}

    if all_retrieval:
        retrieval_df = pd.concat(all_retrieval, ignore_index=True)
        retrieval_df.to_csv(RESULTS_DIR / f"{output_prefix}_retrieval.csv", index=False)
        results["retrieval"] = retrieval_df
        logger.info(f"Saved retrieval results: {len(retrieval_df)} rows")

    if all_generation:
        generation_df = pd.concat(all_generation, ignore_index=True)
        generation_df.to_csv(RESULTS_DIR / f"{output_prefix}_generation.csv", index=False)
        results["generation"] = generation_df
        logger.info(f"Saved generation results: {len(generation_df)} rows")

    if all_faithfulness:
        faith_df = pd.concat(all_faithfulness, ignore_index=True)
        faith_df.to_csv(RESULTS_DIR / f"{output_prefix}_faithfulness.csv", index=False)
        results["faithfulness"] = faith_df
        logger.info(f"Saved faithfulness results: {len(faith_df)} rows")

    return results


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_ablation_results(results: dict[str, pd.DataFrame], output_dir: Optional[Path] = None):
    """
    Generate comparison plots from ablation results.
    """
    output_dir = output_dir or RESULTS_DIR
    sns.set_theme(style="darkgrid", palette="viridis")

    # ── Plot 1: Retrieval Precision@5 by chunking strategy ──
    if "retrieval" in results:
        df = results["retrieval"]

        if "precision@5" in df.columns and "config" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract chunking info from config string
            df["chunk_strategy"] = df["config"].str.extract(r"chunk=(\w+-\d+)")

            pivot = df.groupby("chunk_strategy")["precision@5"].mean()
            pivot.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(pivot)))

            ax.set_title("Retrieval Precision@5 by Chunking Strategy", fontsize=14)
            ax.set_ylabel("Precision@5")
            ax.set_xlabel("Chunking Strategy")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "precision_by_chunking.png", dpi=150)
            plt.close()

    # ── Plot 2: BERTScore by embedding model ──
    if "generation" in results:
        df = results["generation"]

        if "bert_f1" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            df["embedding"] = df["config"].str.extract(r"emb=(\w+)")
            df.groupby("embedding")["bert_f1"].mean().plot(
                kind="bar", ax=ax, color=sns.color_palette("mako", 3)
            )

            ax.set_title("BERTScore F1 by Embedding Model", fontsize=14)
            ax.set_ylabel("BERTScore F1")
            plt.tight_layout()
            plt.savefig(output_dir / "bertscore_by_embedding.png", dpi=150)
            plt.close()

    # ── Plot 3: Faithfulness by LLM ──
    if "faithfulness" in results:
        df = results["faithfulness"]

        if "fact_precision" in df.columns and "llm" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))

            df.groupby("llm")["fact_precision"].mean().plot(
                kind="bar", ax=ax, color=["#4f46e5", "#7c3aed"]
            )

            ax.set_title("Fact Precision by LLM (Fact-Dense Domains)", fontsize=14)
            ax.set_ylabel("Fact Precision")
            plt.tight_layout()
            plt.savefig(output_dir / "faithfulness_by_llm.png", dpi=150)
            plt.close()

    logger.info(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--video-ids", required=True, nargs="+",
        help="Video IDs to evaluate"
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip re-ingestion if indices exist"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Only run retrieval evaluation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick test with 2 configs only"
    )
    parser.add_argument(
        "--output-prefix", default="ablation",
        help="Prefix for output files"
    )
    args = parser.parse_args()

    if args.quick:
        configs = [
            PipelineConfig(
                embedding_model="minilm",
                chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
                retrieval=RetrievalConfig(use_reranker=False),
                generation=GenerationConfig(model_name="mistral"),
            ),
            PipelineConfig(
                embedding_model="minilm",
                chunking=ChunkingConfig(strategy="sentence", chunk_size=500),
                retrieval=RetrievalConfig(use_reranker=True),
                generation=GenerationConfig(model_name="mistral"),
            ),
        ]
    else:
        configs = None  # Full grid

    results = run_ablation(
        video_ids=args.video_ids,
        configs=configs,
        skip_ingestion=args.skip_ingestion,
        skip_generation=args.skip_generation,
        output_prefix=args.output_prefix,
    )

    plot_ablation_results(results)
    print(f"\n✅ Ablation complete. Results saved to {RESULTS_DIR}")
