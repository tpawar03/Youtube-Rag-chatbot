"""
YouTube Video RAG Chatbot — Streamlit Interface

A dark-themed chat application that lets users ask questions
about YouTube video content using a RAG pipeline.
"""

from __future__ import annotations


import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.chat import (
    render_chat_history,
    add_user_message,
    add_assistant_message,
    add_dual_assistant_message,
    format_overview_message,
)
from app.components.evaluation import render_evaluation_page
from src.generation.grounding import score_sentences, render_with_highlights
from app.components.status import render_ingest_complete
from src.pipeline import IngestPipeline, QueryPipeline
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.transcript.fetcher import extract_video_id, TranscriptFetchError
from src.generation.llm import LLMConnectionError


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
# Which model the user has picked as the winner. While None, the first
# question of a turn fans out to both models; once set, subsequent turns
# route only through that model's chain.
if "picked_model" not in st.session_state:
    st.session_state.picked_model = None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
video_url, config, ingest_clicked, ui_state = render_sidebar()
prompt_style = ui_state["prompt_style"]

with st.sidebar:
    st.divider()
    st.subheader("🧭 View")
    view = st.radio(
        "Mode",
        options=["Chat", "Evaluation & Ablation"],
        label_visibility="collapsed",
        key="app_view",
    )

# ─────────────────────────────────────────────
# Evaluation view short-circuits the chat flow
# ─────────────────────────────────────────────
if view == "Evaluation & Ablation":
    render_evaluation_page()
    st.stop()


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

        # Build query pipeline — chains for ALL LLMs are always constructed
        # so the first question can fan out to both Mistral and Llama2.
        query_pipeline = QueryPipeline(
            config=config,
            store=pipeline.store,
            embedder=pipeline.embedder,
            skip_llm_health_check=False,
        )

        # Generate video overview before the user can ask anything.
        # Runs synchronously so the first render already has the overview
        # message seeded — no input-gating flag needed.
        overview_placeholder = st.empty()
        overview_placeholder.info("🧠 Reading the video and preparing an overview...")
        overview = query_pipeline.generate_overview(video_id)
        overview_placeholder.empty()

        # Update session state — new ingest means fresh comparison too.
        st.session_state.current_video_id = video_id
        st.session_state.query_pipeline = query_pipeline
        st.session_state.ingest_info = info
        st.session_state.picked_model = None
        st.session_state.messages = [{
            "role": "assistant",
            "content": format_overview_message(overview),
            "sources": [],
        }]

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

    # Render chat history
    render_chat_history()

    # If there's a dual turn waiting for a pick, block the input so the
    # user can't ask a new question until they've chosen a winner.
    pending_pick = any(
        m.get("dual") and m.get("picked") is None
        for m in st.session_state.messages
    )
    input_placeholder = (
        "Pick a preferred answer above to continue..."
        if pending_pick
        else "Ask a question about the video..."
    )
    prompt = st.chat_input(input_placeholder, disabled=pending_pick)

    if prompt:
        add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        pipeline = st.session_state.query_pipeline
        picked = st.session_state.picked_model

        if pipeline is None:
            with st.chat_message("assistant"):
                st.error("Pipeline not initialized. Please ingest a video first.")
        elif picked is None:
            # No winner yet → fan out to both models on shared context.
            with st.chat_message("assistant"):
                with st.spinner("Asking both Mistral and Llama2..."):
                    try:
                        result = pipeline.ask_dual(prompt, prompt_style=prompt_style)
                        if result.get("off_topic"):
                            # Retrieval gate fired — render a single refusal,
                            # skip the dual pick flow entirely.
                            add_assistant_message(result["answer"])
                        else:
                            turn_id = f"turn_{len(st.session_state.messages)}"
                            add_dual_assistant_message(
                                question=prompt,
                                responses=result["responses"],
                                sources=result["sources"],
                                turn_id=turn_id,
                            )
                    except Exception as e:
                        error_msg = f"Error generating dual response: {e}"
                        st.error(error_msg)
                        add_assistant_message(error_msg)
            st.rerun()
        else:
            # User has already picked a winner → route only to that model.
            with st.chat_message("assistant"):
                with st.spinner(f"Thinking ({picked})..."):
                    try:
                        result = pipeline.ask_with_model(
                            picked, prompt, prompt_style=prompt_style
                        )
                        answer = result.get("answer", "I couldn't generate a response.")
                        sources = result.get("sources", [])
                        candidates = result.get("candidates", [])
                        confidence = result.get("confidence")

                        scored = score_sentences(answer, candidates)
                        highlighted = render_with_highlights(scored)

                        from app.components.chat import render_confidence_badge, render_citations
                        if confidence is not None:
                            st.markdown(
                                f"<div style='margin-bottom: 0.4rem;'>"
                                f"{render_confidence_badge(confidence)}</div>",
                                unsafe_allow_html=True,
                            )
                        # render_with_highlights returns "" when there are
                        # no scored sentences (e.g., off-topic refusal with
                        # no candidates) — fall back to plain text so the
                        # message doesn't disappear.
                        if highlighted:
                            st.markdown(highlighted, unsafe_allow_html=True)
                        else:
                            st.markdown(answer)
                        render_citations(sources)

                        add_assistant_message(
                            answer,
                            sources,
                            grounded_sentences=scored,
                            confidence=confidence,
                        )
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        add_assistant_message(error_msg)
