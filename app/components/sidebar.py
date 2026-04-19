"""
Sidebar component for the Streamlit chat interface.

Provides:
    - YouTube URL input with validation
    - Embedding model selection
    - Chunking strategy and size configuration
    - Cross-encoder reranking toggle
    - Prompt style selector
    - Ingest action button
    - Preference scoreboard across sessions
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
from src.preferences import get_scores


_LLM_LABELS = {"mistral": "Mistral-7B", "llama2": "Llama-2-7B"}


def _render_scoreboard() -> None:
    """Render the dual-model preference scoreboard."""
    scores = get_scores()
    if not scores:
        st.caption("_No picks yet — answer the first question to choose a winner._")
        return
    cols = st.columns(len(LLM_MODELS))
    for col, model_key in zip(cols, LLM_MODELS.keys()):
        label = _LLM_LABELS.get(model_key, model_key)
        with col:
            st.metric(label, scores.get(model_key, 0))


def render_sidebar() -> tuple[str | None, PipelineConfig, bool, dict]:
    """
    Render the sidebar and return the current configuration.

    Returns:
        Tuple of:
            - video_url (str | None): URL entered by user, or None.
            - config (PipelineConfig): Current UI configuration.
            - ingest_clicked (bool): Whether the ingest button was pressed.
            - ui_state (dict): {prompt_style}.
    """
    with st.sidebar:
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

        st.subheader("📺 Video Input")
        video_url = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://youtube.com/watch?v=...",
            key="video_url_input",
        )

        st.divider()

        st.subheader("⚙️ Pipeline Settings")

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

        use_reranker = st.toggle(
            "Cross-Encoder Reranking",
            value=False,
            key="use_reranker",
            help="Re-rank initial FAISS results with a cross-encoder for higher precision.",
        )

        prompt_style = st.selectbox(
            "Prompt Style",
            options=["default", "concise", "detailed"],
            format_func=lambda s: s.capitalize(),
            key="prompt_style",
            help=(
                "Adjusts answer length and depth. Applied to both models on "
                "the first (compare) turn and to the chosen model afterwards."
            ),
        )

        st.caption(
            f"🤖 Every first question runs both **{_LLM_LABELS['mistral']}** "
            f"and **{_LLM_LABELS['llama2']}**. Pick a winner to continue with "
            f"only that model."
        )

        st.divider()

        ingest_clicked = st.button(
            "🚀 Ingest Video",
            use_container_width=True,
            type="primary",
            disabled=not video_url,
        )

        if st.session_state.get("current_video_id"):
            st.divider()
            st.subheader("📊 Current Session")
            st.caption(f"**Video:** `{st.session_state.current_video_id}`")
            if st.session_state.get("ingest_info"):
                info = st.session_state.ingest_info
                st.caption(f"**Chunks:** {info.get('num_chunks', '?')}")
                st.caption(f"**Embed dim:** {info.get('embedding_dim', '?')}")

            picked = st.session_state.get("picked_model")
            if picked:
                st.caption(
                    f"**Active model:** 🏆 {_LLM_LABELS.get(picked, picked)}"
                )
                if st.button(
                    "🔄 Reset model choice",
                    use_container_width=True,
                    help="Clear your pick so the next question runs both models again.",
                ):
                    st.session_state.picked_model = None
                    st.rerun()

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.picked_model = None
                st.rerun()

        st.divider()
        st.subheader("🏆 Preference Scoreboard")
        _render_scoreboard()

    # The LLM Model picker is gone — both models always run on the first
    # turn. GenerationConfig still carries a model_name that QueryPipeline
    # effectively ignores (DUAL_MODELS drives chain construction), so we
    # default it to the first registered model.
    default_llm_key = next(iter(LLM_MODELS.keys()))

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
            model_name=default_llm_key,
        ),
    )

    ui_state = {"prompt_style": prompt_style}

    return (video_url if video_url else None), config, ingest_clicked, ui_state
