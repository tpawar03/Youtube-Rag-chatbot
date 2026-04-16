"""
Chat display component.

Renders conversation messages with user/assistant bubbles
and timestamp citation badges. Supports streaming responses.
"""

import streamlit as st


def render_chat_history():
    """
    Render all messages from session state chat history.
    """
    for message in st.session_state.get("messages", []):
        role = message["role"]
        content = message["content"]
        sources = message.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)

            # Render timestamp citations for assistant messages
            if role == "assistant" and sources:
                render_citations(sources)


def render_citations(sources: list[dict]):
    """
    Render timestamp citation badges below an assistant response.

    Args:
        sources: List of source dicts with start_time, end_time, text.
    """
    if not sources:
        return

    st.markdown(
        "<div style='margin-top: 0.5rem;'>",
        unsafe_allow_html=True,
    )

    # Build citation badges
    badges = []
    for i, src in enumerate(sources, 1):
        start = _seconds_to_mmss(src.get("start_time", 0))
        end = _seconds_to_mmss(src.get("end_time", 0))
        score = src.get("score", 0)

        badge_html = (
            f'<span style="'
            f"display: inline-block; "
            f"background: linear-gradient(135deg, #4f46e5, #7c3aed); "
            f"color: white; "
            f"padding: 0.2rem 0.6rem; "
            f"border-radius: 1rem; "
            f"font-size: 0.75rem; "
            f"margin: 0.15rem; "
            f"font-weight: 500; "
            f"letter-spacing: 0.02em; "
            f'">'
            f"📍 {start} – {end}"
            f"</span>"
        )
        badges.append(badge_html)

    st.markdown(
        " ".join(badges),
        unsafe_allow_html=True,
    )

    # Expandable source details
    with st.expander("📄 View source chunks", expanded=False):
        for i, src in enumerate(sources, 1):
            start = _seconds_to_mmss(src.get("start_time", 0))
            end = _seconds_to_mmss(src.get("end_time", 0))
            text_preview = src.get("text", "")

            st.markdown(
                f"**Chunk {i}** `[{start} – {end}]`\n\n"
                f"> {text_preview}"
            )
            if i < len(sources):
                st.divider()

    st.markdown("</div>", unsafe_allow_html=True)


def add_user_message(content: str):
    """Add a user message to session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "user",
        "content": content,
    })


def add_assistant_message(content: str, sources: list[dict] | None = None):
    """Add an assistant message to session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
    })


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
