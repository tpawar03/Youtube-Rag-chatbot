"""
Regenerate the five plots referenced in evaluation/docs/02_results.md §7.

Reads:
    evaluation/results/full_retrieval_retrieval.csv
    evaluation/results/subset_generation.csv
    evaluation/results/subset_faithfulness.csv

Writes:
    plot_precision_by_chunking.png
    plot_mrr_by_embedding.png
    plot_rerank_effect.png
    plot_bertscore_by_config.png
    plot_faithfulness_by_llm.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR

log = logging.getLogger("plots")


def _style() -> None:
    sns.set_theme(style="darkgrid", palette="viridis")


def plot_precision_by_chunking(df: pd.DataFrame, out: Path) -> None:
    d = df.copy()
    d["chunk"] = d["config"].str.extract(r"chunk=([\w-]+)")
    means = d.groupby("chunk")["precision@5"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(means)))
    ax.set_title("Retrieval Precision@5 by Chunking Strategy")
    ax.set_ylabel("Precision@5")
    ax.set_xlabel("Chunking Strategy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_mrr_by_embedding(df: pd.DataFrame, out: Path) -> None:
    d = df.copy()
    d["embedding"] = d["config"].str.extract(r"emb=(\w+)")
    means = d.groupby("embedding")["mrr"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax, color=sns.color_palette("mako", len(means)))
    ax.set_title("Mean Reciprocal Rank by Embedding Model")
    ax.set_ylabel("MRR")
    ax.set_xlabel("Embedding Model")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_rerank_effect(df: pd.DataFrame, out: Path) -> None:
    d = df.copy()
    d["rerank"] = d["config"].str.extract(r"rerank=(\w+)")

    metrics = ["precision@1", "precision@5", "mrr"]
    means = d.groupby("rerank")[metrics].mean().T
    means = means[["False", "True"]] if set(means.columns) >= {"False", "True"} else means

    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax, color=["#4f46e5", "#c084fc"])
    ax.set_title("Reranker Effect — off vs. on")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.legend(title="Reranker")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_bertscore_by_config(df: pd.DataFrame, out: Path) -> None:
    means = df.groupby("cfg_name")["bert_f1"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    means.plot(kind="bar", ax=ax, color=sns.color_palette("mako", len(means)))
    ax.set_title("BERTScore F1 by Generation Config")
    ax.set_ylabel("BERTScore F1")
    ax.set_xlabel("Config")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_faithfulness_by_llm(df: pd.DataFrame, out: Path) -> None:
    means = df.groupby("llm")["fact_precision"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    means.plot(kind="bar", ax=ax, color=["#4f46e5", "#7c3aed"])
    ax.set_title("Fact Precision by LLM (Fact-Dense Subset)")
    ax.set_ylabel("Fact Precision")
    ax.set_xlabel("LLM")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _style()

    retrieval_csv = RESULTS_DIR / "full_retrieval_retrieval.csv"
    gen_csv = RESULTS_DIR / "subset_generation.csv"
    faith_csv = RESULTS_DIR / "subset_faithfulness.csv"

    if retrieval_csv.exists():
        ret = pd.read_csv(retrieval_csv)
        plot_precision_by_chunking(ret, RESULTS_DIR / "plot_precision_by_chunking.png")
        plot_mrr_by_embedding(ret, RESULTS_DIR / "plot_mrr_by_embedding.png")
        plot_rerank_effect(ret, RESULTS_DIR / "plot_rerank_effect.png")
        log.info("Retrieval plots written")
    else:
        log.warning("Missing %s — skipping retrieval plots", retrieval_csv)

    if gen_csv.exists():
        gen = pd.read_csv(gen_csv)
        plot_bertscore_by_config(gen, RESULTS_DIR / "plot_bertscore_by_config.png")
        log.info("BERTScore plot written")
    else:
        log.warning("Missing %s — skipping generation plot", gen_csv)

    if faith_csv.exists():
        faith = pd.read_csv(faith_csv)
        plot_faithfulness_by_llm(faith, RESULTS_DIR / "plot_faithfulness_by_llm.png")
        log.info("Faithfulness plot written")
    else:
        log.warning("Missing %s — skipping faithfulness plot", faith_csv)


if __name__ == "__main__":
    main()
