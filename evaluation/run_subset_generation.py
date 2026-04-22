"""
Driver for the 6-config generation + faithfulness subset.

Pulled verbatim from evaluation/docs/02_results.md §9 so the run is
re-executable. Produces:
    - evaluation/results/subset_generation.csv
    - evaluation/results/subset_faithfulness.csv
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from config import (
    PipelineConfig,
    ChunkingConfig,
    RetrievalConfig,
    GenerationConfig,
    RESULTS_DIR,
)
from src.pipeline import build_query_pipeline
from evaluation.retrieval_eval import load_qa_pairs
from evaluation.generation_eval import evaluate_generation
from evaluation.faithfulness_eval import compute_fact_precision


CONFIGS = {
    "top-minilm-mistral": PipelineConfig(
        embedding_model="minilm",
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name="mistral"),
    ),
    "top-minilm-llama2": PipelineConfig(
        embedding_model="minilm",
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name="llama2"),
    ),
    "top-mpnet-mistral": PipelineConfig(
        embedding_model="mpnet",
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name="mistral"),
    ),
    "top-e5-mistral": PipelineConfig(
        embedding_model="e5",
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name="mistral"),
    ),
    "sentence-mistral": PipelineConfig(
        embedding_model="minilm",
        chunking=ChunkingConfig(strategy="sentence", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name="mistral"),
    ),
    "no-rerank-mistral": PipelineConfig(
        embedding_model="minilm",
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=False),
        generation=GenerationConfig(model_name="mistral"),
    ),
}

VIDEO_IDS = ["VSFuqMh4hus", "RRKwmeyIc24"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("subset_gen")

    qa_pairs = load_qa_pairs()
    all_gen: list[pd.DataFrame] = []
    all_faith: list[dict] = []

    t_start = time.time()
    total = len(CONFIGS) * len(VIDEO_IDS)
    step = 0

    for cfg_name, cfg in CONFIGS.items():
        for video_id in VIDEO_IDS:
            step += 1
            video_qa = [q for q in qa_pairs if q.get("video_id") == video_id]
            if not video_qa:
                log.warning("No QA pairs for %s — skipping", video_id)
                continue

            log.info("[%d/%d] %s × %s (%d QA)", step, total, cfg_name, video_id, len(video_qa))
            t_cfg = time.time()

            qp = build_query_pipeline(video_id, cfg, skip_llm_health_check=True)

            generated: list[str] = []
            contexts: list[list[str]] = []
            for qa in video_qa:
                qp.reset()
                chunks = qp.retrieve_only(qa["question"])
                contexts.append([c["text"] for c in chunks])
                generated.append(qp.ask(qa["question"])["answer"])

            gen_df = evaluate_generation(cfg, video_qa, generated)
            gen_df["cfg_name"] = cfg_name
            all_gen.append(gen_df)

            for qa, ans, ctx in zip(video_qa, generated, contexts):
                fp = compute_fact_precision(ans, " ".join(ctx))
                all_faith.append({
                    "cfg_name": cfg_name,
                    "video_id": video_id,
                    "domain": qa.get("domain"),
                    "question": qa["question"],
                    "num_facts": fp["num_facts"],
                    "supported_facts": fp["supported_facts"],
                    "fact_precision": fp["fact_precision"],
                    "llm": cfg.generation.model_name,
                })

            log.info("  done in %.1fs", time.time() - t_cfg)

    gen_out = RESULTS_DIR / "subset_generation.csv"
    faith_out = RESULTS_DIR / "subset_faithfulness.csv"

    pd.concat(all_gen, ignore_index=True).to_csv(gen_out, index=False)
    pd.DataFrame(all_faith).to_csv(faith_out, index=False)

    log.info("Wrote %s (%d rows)", gen_out, sum(len(d) for d in all_gen))
    log.info("Wrote %s (%d rows)", faith_out, len(all_faith))
    log.info("Total wall time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
