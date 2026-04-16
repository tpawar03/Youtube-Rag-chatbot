"""
Retrieval quality evaluation.

Computes Precision@K, Recall@K, Hit Rate, and MRR against
a hand-annotated set of question-answer pairs with labeled
relevant chunk IDs.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PipelineConfig, EVAL_DIR, RESULTS_DIR
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


def load_qa_pairs(path: Optional[Path] = None) -> list[dict]:
    """Load annotated QA pairs from JSON."""
    path = path or EVAL_DIR / "dataset" / "qa_pairs.json"
    with open(path, "r") as f:
        return json.load(f)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of top-K retrieved that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def hit_rate(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """1 if any relevant item is in top-K, else 0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Reciprocal of the rank of the first relevant item."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    config: PipelineConfig,
    qa_pairs: list[dict],
    video_id: str,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate retrieval quality for a given configuration.

    Args:
        config: Pipeline configuration.
        qa_pairs: List of QA pair dicts with keys:
            - question (str)
            - answer (str)
            - relevant_chunk_ids (list[int])
            - video_id (str)
            - domain (str)
        video_id: YouTube video ID.
        k_values: List of K values: defaults to [1, 3, 5, 10].

    Returns:
        DataFrame with per-question and aggregate metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Load index
    embedder = Embedder(config.embedding_model)
    store = FAISSStore(embedder)
    index_name = f"{video_id}_{config.index_id}"

    try:
        store.load(index_name)
    except FileNotFoundError:
        logger.error(f"Index not found: {index_name}. Run ingestion first.")
        return pd.DataFrame()

    # Setup retriever
    retriever = Retriever(store, config.retrieval)
    reranker = Reranker(config.retrieval) if config.retrieval.use_reranker else None

    # Filter QA pairs for this video
    video_qa = [qa for qa in qa_pairs if qa.get("video_id") == video_id]

    if not video_qa:
        logger.warning(f"No QA pairs found for video {video_id}")
        return pd.DataFrame()

    results = []

    for qa in video_qa:
        question = qa["question"]
        relevant_ids = set(qa.get("relevant_chunk_ids", []))
        domain = qa.get("domain", "unknown")

        # Retrieve
        candidates = retriever.retrieve(question)

        if reranker and config.retrieval.use_reranker:
            candidates = reranker.rerank(
                question, candidates, top_k=max(k_values)
            )

        retrieved_ids = [c.get("chunk_index", -1) for c in candidates]

        # Compute metrics for each K
        row = {
            "question": question,
            "domain": domain,
            "video_id": video_id,
            "config": str(config),
        }

        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            row[f"hit_rate@{k}"] = hit_rate(retrieved_ids, relevant_ids, k)

        row["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)

        results.append(row)

    df = pd.DataFrame(results)
    return df


def aggregate_retrieval_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate retrieval results by domain and overall.

    Args:
        df: Per-question DataFrame from evaluate_retrieval.

    Returns:
        Aggregated DataFrame with mean metrics per domain + overall.
    """
    if df.empty:
        return df

    metric_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["precision@", "recall@", "hit_rate@", "mrr"]
    )]

    # Per-domain aggregation
    domain_agg = df.groupby("domain")[metric_cols].mean()
    domain_agg.index.name = "group"

    # Overall aggregation
    overall = df[metric_cols].mean().to_frame().T
    overall.index = ["overall"]
    overall.index.name = "group"

    return pd.concat([domain_agg, overall])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--video-id", required=True, help="YouTube video ID")
    parser.add_argument("--embedding", default="minilm", help="Embedding model key")
    parser.add_argument("--chunk-strategy", default="fixed", help="fixed or sentence")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    from config import ChunkingConfig, RetrievalConfig

    cfg = PipelineConfig(
        embedding_model=args.embedding,
        chunking=ChunkingConfig(strategy=args.chunk_strategy, chunk_size=args.chunk_size),
        retrieval=RetrievalConfig(use_reranker=args.reranker),
    )

    qa_pairs = load_qa_pairs()
    results = evaluate_retrieval(cfg, qa_pairs, args.video_id)

    if not results.empty:
        agg = aggregate_retrieval_results(results)
        print("\n=== Retrieval Evaluation Results ===")
        print(agg.to_string())

        output_path = args.output or str(RESULTS_DIR / f"retrieval_{cfg.index_id}.csv")
        results.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
