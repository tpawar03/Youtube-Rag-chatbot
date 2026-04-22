"""
Ingestion progress status component.

Displays pipeline stage progress during video ingestion.
"""

from __future__ import annotations


import streamlit as st


def render_ingestion_status(stage: str, progress: float):
    """
    Update the ingestion progress display.

    Args:
        stage: Current pipeline stage name.
        progress: Progress fraction (0.0 to 1.0).
    """
    stages = [
        ("Fetching transcript", "📥"),
        ("Cleaning transcript", "🧹"),
        ("Chunking transcript", "✂️"),
        ("Embedding chunks", "🧮"),
        ("Saving index", "💾"),
        ("Complete", "✅"),
    ]

    # Progress bar
    st.progress(progress, text=f"{stage}...")

    # Stage indicators
    cols = st.columns(len(stages))
    for i, (stage_name, emoji) in enumerate(stages):
        with cols[i]:
            stage_progress = (i + 1) / len(stages)
            if progress >= stage_progress:
                st.markdown(
                    f"<div style='text-align:center; color:#22c55e;'>"
                    f"{emoji}<br><small>{stage_name.split(' ')[0]}</small></div>",
                    unsafe_allow_html=True,
                )
            elif progress >= stage_progress - 1 / len(stages):
                st.markdown(
                    f"<div style='text-align:center; color:#818cf8;'>"
                    f"⏳<br><small>{stage_name.split(' ')[0]}</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; color:#475569;'>"
                    f"⬜<br><small>{stage_name.split(' ')[0]}</small></div>",
                    unsafe_allow_html=True,
                )


def render_ingest_complete(info: dict):
    """
    Show summary after successful ingestion.

    Args:
        info: Dict from IngestPipeline.ingest() with video stats.
    """
    st.success("✅ Video ingested successfully!", icon="🎉")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Segments", info.get("num_segments", "?"))
    with col2:
        st.metric("Chunks", info.get("num_chunks", "?"))
    with col3:
        st.metric("Embed Dim", info.get("embedding_dim", "?"))
