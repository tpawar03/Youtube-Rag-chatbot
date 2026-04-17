"""
YouTube Video RAG Chatbot — Streamlit Interface

A dark-themed chat application that lets users ask questions
about YouTube video content using a RAG pipeline.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.chat import (
    render_chat_history,
    render_dual_comparison,
    add_user_message,
    add_assistant_message,
)
from app.components.status import render_ingest_complete
from src.pipeline import IngestPipeline, QueryPipeline
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.transcript.fetcher import extract_video_id, TranscriptFetchError
from src.generation.llm import LLMConnectionError
from app.utils import log_preference


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VideoChat AI — YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark theme with glassmorphism
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main chat area background */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 900px;
    }

    /* Header styling */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero-header h1 {
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 1rem !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 1rem !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
    }
    .stChatInput > div:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 20px rgba(129, 140, 248, 0.15) !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border: none !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4) !important;
    }

    /* Metrics */
    .stMetric {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 0.75rem;
        padding: 0.75rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* Status/Progress */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #c084fc) !important;
        border-radius: 1rem !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 0.5rem !important;
    }

    /* Toggle */
    .stToggle label span {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "query_pipeline" not in st.session_state:
    st.session_state.query_pipeline = None
if "ingest_info" not in st.session_state:
    st.session_state.ingest_info = None
if "pending_dual" not in st.session_state:
    st.session_state.pending_dual = None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
video_url, config, ingest_clicked, prompt_style = render_sidebar()


# ─────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────

# Hero header (shown when no video is loaded)
if not st.session_state.current_video_id:
    st.markdown(
        """
        <div class="hero-header">
            <h1>VideoChat AI</h1>
            <p>Ask questions about any YouTube video · Powered by RAG</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            color: #64748b;
            font-size: 0.95rem;
            line-height: 1.8;
        ">
            <p>👈 Paste a YouTube URL in the sidebar and click <strong>Ingest Video</strong></p>
            <p style="margin-top: 1rem;">
                📺 Fetches transcript &nbsp;→&nbsp;
                ✂️ Chunks text &nbsp;→&nbsp;
                🧮 Embeds vectors &nbsp;→&nbsp;
                💬 Chat!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Ingestion handler
# ─────────────────────────────────────────────
if ingest_clicked and video_url:
    try:
        video_id = extract_video_id(video_url)

        # Progress container
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with st.spinner(""):
            # Create progress callback
            def update_progress(stage: str, progress: float):
                progress_placeholder.progress(
                    progress,
                    text=f"{'🔄' if progress < 1.0 else '✅'} {stage}",
                )

            # Run ingestion
            pipeline = IngestPipeline(
                config=config,
                progress_callback=update_progress,
            )
            info = pipeline.ingest(video_url)

        # Clear progress indicators
        progress_placeholder.empty()

        # Show completion
        with status_placeholder.container():
            render_ingest_complete(info)

        # Build query pipeline
        query_pipeline = QueryPipeline(
            config=config,
            store=pipeline.store,
            embedder=pipeline.embedder,
            skip_llm_health_check=False,
        )

        # Update session state
        st.session_state.current_video_id = video_id
        st.session_state.query_pipeline = query_pipeline
        st.session_state.ingest_info = info
        st.session_state.messages = []

        st.rerun()

    except TranscriptFetchError as e:
        st.error(f"❌ Transcript Error: {e}")
    except LLMConnectionError as e:
        st.error(f"❌ LLM Error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        raise


# ─────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────
if st.session_state.current_video_id:
    # Show loaded video info
    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1.25rem;
            background: linear-gradient(135deg, rgba(79,70,229,0.1), rgba(124,58,237,0.1));
            border: 1px solid rgba(129,140,248,0.2);
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="color: #c4b5fd; font-size: 0.9rem;">
                🎬 <strong>{st.session_state.current_video_id}</strong>
            </span>
            <span style="color: #94a3b8; font-size: 0.8rem;">
                {st.session_state.ingest_info.get('num_chunks', '?')} chunks indexed
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Warn if second model is unavailable
    pipeline = st.session_state.query_pipeline
    if pipeline and pipeline.secondary_chain is None:
        st.warning(
            f"Second model **{pipeline._secondary_model_key}** is not available in Ollama. "
            f"Run `ollama pull {pipeline._secondary_model_key}` to enable dual-model comparison. "
            f"Falling back to single-model mode.",
            icon="⚠️",
        )

    # Render chat history
    render_chat_history()

    # ── Dual-model comparison (pending selection) ──
    if st.session_state.pending_dual is not None:
        pending = st.session_state.pending_dual
        selected_answer = render_dual_comparison(pending)

        if selected_answer is not None:
            if selected_answer == pending["primary"]["answer"]:
                selected = pending["primary"]
                loser_model = pending["secondary"]["model"]
            else:
                selected = pending["secondary"]
                loser_model = pending["primary"]["model"]

            log_preference(
                winner=selected["model"],
                loser=loser_model,
                question=pending["question"],
                video_id=st.session_state.current_video_id or "",
            )
            pipeline.commit_dual_selection(pending["question"], selected_answer)
            add_assistant_message(
                selected_answer,
                selected["sources"],
                confidence=selected.get("confidence"),
                sentence_scores=selected.get("sentence_scores", []),
            )
            st.session_state.pending_dual = None
            st.rerun()

    # Chat input (disabled while awaiting selection)
    elif prompt := st.chat_input("Ask a question about the video..."):
        add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        if pipeline is None:
            st.error("Pipeline not initialized. Please ingest a video first.")
        elif pipeline.secondary_chain is None:
            # Fallback: single-model response
            with st.spinner("Thinking..."):
                try:
                    result = pipeline.ask(prompt, prompt_style=prompt_style)
                    answer = result.get("answer", "I couldn't generate a response.")
                    sources = result.get("sources", [])
                    confidence = result.get("confidence")
                    sentence_scores = result.get("sentence_scores", [])
                    with st.chat_message("assistant"):
                        from app.components.chat import render_highlighted_answer, render_confidence, render_citations
                        if sentence_scores:
                            render_highlighted_answer(sentence_scores)
                        else:
                            st.markdown(answer)
                        render_confidence(confidence)
                        render_citations(sources)
                    add_assistant_message(answer, sources, confidence, sentence_scores)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            with st.spinner("Generating responses from both models..."):
                try:
                    dual_result = pipeline.ask_dual(prompt, prompt_style=prompt_style)
                    st.session_state.pending_dual = dual_result
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {e}")
