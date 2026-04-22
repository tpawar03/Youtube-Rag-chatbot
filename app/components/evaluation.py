"""
Evaluation & ablation results viewer.

Reads the CSVs and PNGs under ``evaluation/results/`` and renders them
as aggregate tables and plots inside the Streamlit app so users can
inspect the study without leaving the chat interface.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import RESULTS_DIR


RETRIEVAL_CSV = RESULTS_DIR / "full_retrieval_retrieval.csv"
GENERATION_CSV = RESULTS_DIR / "subset_generation.csv"
FAITHFULNESS_CSV = RESULTS_DIR / "subset_faithfulness.csv"

PLOTS = {
    "Precision@5 by chunking": "plot_precision_by_chunking.png",
    "MRR by embedding": "plot_mrr_by_embedding.png",
    "Reranker on vs. off": "plot_rerank_effect.png",
    "BERTScore by generation config": "plot_bertscore_by_config.png",
    "Faithfulness by LLM": "plot_faithfulness_by_llm.png",
}


@st.cache_data(show_spinner=False)
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _missing(label: str, path: Path) -> None:
    st.info(
        f"**{label}** not found at `{path.relative_to(RESULTS_DIR.parent.parent)}`. "
        "Run the evaluation pipeline to generate it."
    )


def _extract_axis(df: pd.DataFrame, pattern: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[name] = df["config"].str.extract(pattern)
    return df


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
def _render_retrieval(df: pd.DataFrame) -> None:
    metric_cols = [
        "precision@1", "precision@3", "precision@5",
        "recall@5", "hit_rate@5", "mrr",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Configs", df["config"].nunique() if "config" in df.columns else "—")
    c3.metric("Videos", df["video_id"].nunique() if "video_id" in df.columns else "—")
    if "mrr" in df.columns:
        c4.metric("Mean MRR", f"{df['mrr'].mean():.3f}")

    st.markdown("##### Overall means")
    st.dataframe(
        df[metric_cols].mean().to_frame("value").T.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("##### By chunking strategy")
    d = _extract_axis(df, r"chunk=([\w-]+)", "chunk")
    st.dataframe(
        d.groupby("chunk")[metric_cols].mean().style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("##### By embedding model")
    d = _extract_axis(df, r"emb=(\w+)", "embedding")
    st.dataframe(
        d.groupby("embedding")[metric_cols].mean().style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("##### Reranker on vs. off")
    d = _extract_axis(df, r"rerank=(\w+)", "rerank")
    st.dataframe(
        d.groupby("rerank")[metric_cols].mean().style.format("{:.3f}"),
        use_container_width=True,
    )

    if "domain" in df.columns and "video_id" in df.columns:
        st.markdown("##### By domain / video")
        st.dataframe(
            df.groupby(["domain", "video_id"])[metric_cols]
              .mean()
              .style.format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("🔍 Raw rows (first 200)"):
        st.dataframe(df.head(200), use_container_width=True)


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
def _render_generation(df: pd.DataFrame) -> None:
    metric_cols = [
        "bleu", "rouge1", "rouge2", "rougeL",
        "bert_precision", "bert_recall", "bert_f1",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Configs", df["cfg_name"].nunique() if "cfg_name" in df.columns else "—")
    if "bert_f1" in df.columns:
        c3.metric("Mean BERT-F1", f"{df['bert_f1'].mean():.3f}")

    if "cfg_name" in df.columns:
        st.markdown("##### By config")
        st.dataframe(
            df.groupby("cfg_name")[metric_cols]
              .mean()
              .sort_values("bert_f1" if "bert_f1" in metric_cols else metric_cols[0],
                           ascending=False)
              .style.format("{:.3f}"),
            use_container_width=True,
        )

    if "domain" in df.columns:
        st.markdown("##### By domain")
        st.dataframe(
            df.groupby("domain")[metric_cols].mean().style.format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("🔍 Raw rows (first 200)"):
        st.dataframe(df.head(200), use_container_width=True)


# ──────────────────────────────────────────────
# Faithfulness
# ──────────────────────────────────────────────
def _render_faithfulness(df: pd.DataFrame) -> None:
    metric_cols = [c for c in ["num_facts", "supported_facts", "fact_precision"] if c in df.columns]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    if "fact_precision" in df.columns:
        c2.metric("Mean fact precision", f"{df['fact_precision'].mean():.3f}")
    if "llm" in df.columns:
        c3.metric("LLMs", df["llm"].nunique())

    if "cfg_name" in df.columns:
        st.markdown("##### By config")
        st.dataframe(
            df.groupby("cfg_name")[metric_cols]
              .mean()
              .sort_values("fact_precision", ascending=False)
              .style.format("{:.3f}"),
            use_container_width=True,
        )

    if "llm" in df.columns:
        st.markdown("##### By LLM")
        st.dataframe(
            df.groupby("llm")[metric_cols].mean().style.format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("🔍 Raw rows (first 200)"):
        st.dataframe(df.head(200), use_container_width=True)


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
def _render_plots() -> None:
    found_any = False
    for label, fname in PLOTS.items():
        path = RESULTS_DIR / fname
        if path.exists():
            found_any = True
            st.markdown(f"**{label}**")
            st.image(str(path), use_container_width=True)
    if not found_any:
        st.info("No plot PNGs found. Run `python -m evaluation.plots`.")


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────
def render_evaluation_page() -> None:
    """Render the full Evaluation & Ablation view."""
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem;">
          <h2 style="margin:0; color:#c4b5fd;">📊 Evaluation &amp; Ablation</h2>
          <p style="color:#94a3b8; font-size:0.9rem; margin-top:0.2rem;">
            Results from the 48-config retrieval ablation and 6-config
            generation + faithfulness subset. Source CSVs and PNGs live
            under <code>evaluation/results/</code>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    retrieval = _read_csv(RETRIEVAL_CSV)
    generation = _read_csv(GENERATION_CSV)
    faithfulness = _read_csv(FAITHFULNESS_CSV)

    if st.button("🔄 Reload results", help="Re-read CSVs from disk"):
        _read_csv.clear()
        st.rerun()

    tab_ret, tab_gen, tab_faith, tab_plots = st.tabs(
        ["Retrieval", "Generation", "Faithfulness", "Plots"]
    )

    with tab_ret:
        st.caption(f"Source: `{RETRIEVAL_CSV.name}`")
        if retrieval.empty:
            _missing("Retrieval results", RETRIEVAL_CSV)
        else:
            _render_retrieval(retrieval)

    with tab_gen:
        st.caption(f"Source: `{GENERATION_CSV.name}`")
        if generation.empty:
            _missing("Generation results", GENERATION_CSV)
        else:
            _render_generation(generation)

    with tab_faith:
        st.caption(f"Source: `{FAITHFULNESS_CSV.name}`")
        if faithfulness.empty:
            _missing("Faithfulness results", FAITHFULNESS_CSV)
        else:
            _render_faithfulness(faithfulness)

    with tab_plots:
        _render_plots()
