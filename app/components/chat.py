"""
Chat display component.

Renders conversation messages with user/assistant bubbles
and timestamp citation badges. Supports streaming responses.
"""

import streamlit as st


def render_confidence(confidence: int | None):
    """Render a coloured confidence badge (1-5 scale)."""
    if confidence is None:
        return
    colors = {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#22c55e", 5: "#10b981"}
    labels = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}
    color = colors.get(confidence, "#94a3b8")
    label = labels.get(confidence, "")
    st.markdown(
        f"<div style='margin-top:0.4rem;'>"
        f"<span style='background:{color}22;border:1px solid {color}66;"
        f"color:{color};padding:0.15rem 0.55rem;border-radius:1rem;"
        f"font-size:0.75rem;font-weight:600;'>"
        f"Confidence: {confidence}/5 — {label}"
        f"</span></div>",
        unsafe_allow_html=True,
    )



def render_highlighted_answer(sentence_scores: list[dict]):
    """Render answer with ungrounded sentences underlined in amber."""
    if not sentence_scores:
        return
    parts = []
    for s in sentence_scores:
        text = s["sentence"].replace("<", "&lt;").replace(">", "&gt;")
        if s["grounded"]:
            parts.append(text)
        else:
            score_pct = f"{s['score']:.0%}"
            parts.append(
                f"<span style='border-bottom:2px solid #fbbf24;' "
                f"title='Possibly not grounded in transcript (overlap: {score_pct})'>"
                f"{text}</span>"
            )
    st.markdown(" ".join(parts), unsafe_allow_html=True)
    st.caption("⚠️ Underlined text may not be directly supported by the transcript.")


def render_chat_history():
    """Render all messages from session state chat history."""
    for message in st.session_state.get("messages", []):
        role = message["role"]
        content = message["content"]
        sources = message.get("sources", [])
        confidence = message.get("confidence")
        sentence_scores = message.get("sentence_scores", [])

        with st.chat_message(role):
            if role == "assistant" and sentence_scores:
                render_highlighted_answer(sentence_scores)
            else:
                st.markdown(content)

            if role == "assistant":
                render_confidence(confidence)
                if sources:
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


def render_dual_comparison(pending: dict) -> str | None:
    """
    Render side-by-side model responses and return the selected answer,
    or None if the user hasn't chosen yet.

    Args:
        pending: Dict from QueryPipeline.ask_dual() with keys:
            question, primary, secondary, candidates.

    Returns:
        The selected answer string, or None.
    """
    primary = pending["primary"]
    secondary = pending["secondary"]

    model_labels = {
        "mistral": "Mistral-7B",
        "llama2": "Llama-2-7B",
    }

    st.markdown(
        "<div style='margin: 0.5rem 0 1rem; color: #94a3b8; font-size: 0.85rem;'>"
        "Two models answered your question. Pick the better response:"
        "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    selected = None

    with col_a:
        label_a = model_labels.get(primary["model"], primary["model"])
        st.markdown(
            f"<div style='"
            f"background: rgba(79,70,229,0.08);"
            f"border: 1px solid rgba(129,140,248,0.3);"
            f"border-radius: 0.75rem;"
            f"padding: 1rem;"
            f"margin-bottom: 0.75rem;"
            f"'>"
            f"<span style='color:#818cf8;font-weight:600;font-size:0.8rem;'>"
            f"Model A — {label_a}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if primary.get("sentence_scores"):
            render_highlighted_answer(primary["sentence_scores"])
        else:
            st.markdown(primary["answer"])
        render_confidence(primary.get("confidence"))
        render_citations(primary["sources"])
        if st.button("✓ Choose Model A", key="select_primary", use_container_width=True, type="primary"):
            selected = primary["answer"]

    with col_b:
        label_b = model_labels.get(secondary["model"], secondary["model"])
        st.markdown(
            f"<div style='"
            f"background: rgba(192,132,252,0.08);"
            f"border: 1px solid rgba(192,132,252,0.3);"
            f"border-radius: 0.75rem;"
            f"padding: 1rem;"
            f"margin-bottom: 0.75rem;"
            f"'>"
            f"<span style='color:#c084fc;font-weight:600;font-size:0.8rem;'>"
            f"Model B — {label_b}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if secondary.get("sentence_scores"):
            render_highlighted_answer(secondary["sentence_scores"])
        else:
            st.markdown(secondary["answer"])
        render_confidence(secondary.get("confidence"))
        render_citations(secondary["sources"])
        if st.button("✓ Choose Model B", key="select_secondary", use_container_width=True, type="primary"):
            selected = secondary["answer"]

    return selected


def add_user_message(content: str):
    """Add a user message to session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "user",
        "content": content,
    })


def add_assistant_message(
    content: str,
    sources: list[dict] | None = None,
    confidence: int | None = None,
    sentence_scores: list[dict] | None = None,
):
    """Add an assistant message to session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
        "confidence": confidence,
        "sentence_scores": sentence_scores or [],
    })


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
