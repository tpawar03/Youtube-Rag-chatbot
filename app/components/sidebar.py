"""
Sidebar component for the Streamlit chat interface.

Provides:
    - YouTube URL input with validation
    - Embedding model selection
    - Chunking strategy and size configuration
    - Cross-encoder reranking toggle
    - LLM model selection
    - Ingest action button
"""

import streamlit as st

from config import (
    PipelineConfig,
    ChunkingConfig,
    RetrievalConfig,
    GenerationConfig,
    EMBEDDING_MODELS,
    LLM_MODELS,
)
from src.generation.prompts import PROMPT_STYLE_LABELS
from app.utils import load_scoreboard


def render_sidebar() -> tuple[str | None, PipelineConfig, bool, str]:
    """
    Render the sidebar and return the current configuration.

    Returns:
        Tuple of:
            - video_url (str | None): URL entered by user, or None.
            - config (PipelineConfig): Current UI configuration.
            - ingest_clicked (bool): Whether the ingest button was pressed.
            - prompt_style (str): Selected prompt style key.
    """
    with st.sidebar:
        # ── Header ──
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="margin: 0; color: #818cf8;">🎬 VideoChat AI</h2>
                <p style="color: #94a3b8; font-size: 0.85rem;">
                    RAG-powered YouTube Q&A
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Video URL Input ──
        st.subheader("📺 Video Input")
        video_url = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://youtube.com/watch?v=...",
            key="video_url_input",
        )

        st.divider()

        # ── Model Settings ──
        st.subheader("⚙️ Pipeline Settings")

        # Embedding model
        embedding_key = st.selectbox(
            "Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda k: {
                "minilm": "MiniLM-L6-v2 (Fast)",
                "mpnet": "MPNet-base-v2 (Balanced)",
                "e5": "E5-small-v2 (Instruction-tuned)",
            }.get(k, k),
            key="embedding_model",
        )

        # Chunking strategy
        col1, col2 = st.columns(2)
        with col1:
            chunk_strategy = st.selectbox(
                "Chunking",
                options=["fixed", "sentence"],
                format_func=lambda s: s.capitalize(),
                key="chunk_strategy",
            )
        with col2:
            chunk_size = st.selectbox(
                "Chunk Size",
                options=[200, 500, 1000],
                index=1,
                key="chunk_size",
            )

        # Reranking
        use_reranker = st.toggle(
            "Cross-Encoder Reranking",
            value=False,
            key="use_reranker",
            help="Re-rank initial FAISS results with a cross-encoder for higher precision.",
        )

        # LLM selection
        llm_key = st.selectbox(
            "LLM Model",
            options=list(LLM_MODELS.keys()),
            format_func=lambda k: {
                "mistral": "Mistral-7B",
                "llama2": "Llama-2-7B",
            }.get(k, k),
            key="llm_model",
        )

        # Prompt style
        prompt_style = st.selectbox(
            "Response Style",
            options=list(PROMPT_STYLE_LABELS.keys()),
            format_func=lambda k: PROMPT_STYLE_LABELS.get(k, k),
            key="prompt_style",
            help="Controls how the LLM structures its answer.",
        )

        st.divider()

        # ── Ingest Button ──
        ingest_clicked = st.button(
            "🚀 Ingest Video",
            use_container_width=True,
            type="primary",
            disabled=not video_url,
        )

        # ── Current Config Display ──
        if st.session_state.get("current_video_id"):
            st.divider()
            st.subheader("📊 Current Session")
            st.caption(f"**Video:** `{st.session_state.current_video_id}`")
            if st.session_state.get("ingest_info"):
                info = st.session_state.ingest_info
                st.caption(f"**Chunks:** {info.get('num_chunks', '?')}")
                st.caption(f"**Embed dim:** {info.get('embedding_dim', '?')}")

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # ── Model Scoreboard ──
        scoreboard = load_scoreboard()
        if scoreboard:
            st.divider()
            st.subheader("🏆 Model Scoreboard")
            model_labels = {"mistral": "Mistral-7B", "llama2": "Llama-2-7B"}
            total = sum(scoreboard.values())
            for model, wins in sorted(scoreboard.items(), key=lambda x: -x[1]):
                label = model_labels.get(model, model)
                pct = int(wins / total * 100) if total else 0
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:0.3rem;'>"
                    f"<span style='color:#c4b5fd;font-size:0.85rem;'>{label}</span>"
                    f"<span style='color:#94a3b8;font-size:0.85rem;'>{wins} wins ({pct}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.progress(pct / 100)

    # ── Build config ──
    config = PipelineConfig(
        embedding_model=embedding_key,
        chunking=ChunkingConfig(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
        ),
        retrieval=RetrievalConfig(
            use_reranker=use_reranker,
        ),
        generation=GenerationConfig(
            model_name=llm_key,
        ),
    )

    return (video_url if video_url else None), config, ingest_clicked, prompt_style
