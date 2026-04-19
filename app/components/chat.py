"""
Chat display component.

Renders conversation messages with user/assistant bubbles
and timestamp citation badges. Supports streaming responses.
"""

import streamlit as st

from src.generation.grounding import render_with_highlights
from src.preferences import record_preference


_LLM_LABELS = {"mistral": "Mistral-7B", "llama2": "Llama-2-7B"}


def render_confidence_badge(value: int | None) -> str:
    """
    Return an HTML badge for a 1-5 LLM self-rated confidence score.
    Returns empty string when value is None so callers can unconditionally
    concatenate without a branch.
    """
    if value is None:
        return ""
    if value <= 2:
        color = "#ef4444"  # red
    elif value == 3:
        color = "#f59e0b"  # amber
    else:
        color = "#10b981"  # green
    return (
        f'<span style="'
        f"display: inline-block; "
        f"background: {color}; "
        f"color: white; "
        f"padding: 0.15rem 0.55rem; "
        f"border-radius: 0.75rem; "
        f"font-size: 0.7rem; "
        f"font-weight: 600; "
        f"letter-spacing: 0.03em; "
        f"margin-left: 0.4rem; "
        f"vertical-align: middle;"
        f'" title="LLM self-rated confidence">'
        f"Confidence {value}/5"
        f"</span>"
    )


def render_chat_history():
    """
    Render all messages from session state chat history.
    Supports two message shapes:
      - single: {role, content, sources?}
      - dual:   {role:"assistant", dual:True, responses:{...}, sources:[],
                  question:str, picked:str|None, turn_id:str}
    """
    for idx, message in enumerate(st.session_state.get("messages", [])):
        role = message["role"]

        if role == "assistant" and message.get("dual"):
            with st.chat_message("assistant"):
                render_dual_response(message, message_index=idx)
            continue

        content = message.get("content", "")
        sources = message.get("sources", [])
        scored = message.get("grounded_sentences", []) if role == "assistant" else []
        confidence = message.get("confidence") if role == "assistant" else None

        with st.chat_message(role):
            if confidence is not None:
                st.markdown(
                    f"<div style='margin-bottom: 0.4rem;'>"
                    f"{render_confidence_badge(confidence)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if scored:
                st.markdown(render_with_highlights(scored), unsafe_allow_html=True)
            else:
                st.markdown(content)
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


def render_dual_response(message: dict, message_index: int) -> None:
    """
    Render a dual-model assistant turn as two side-by-side columns with
    hallucination-highlighted answers, citations, and pick buttons.

    The message dict is mutated in place when the user picks a winner, so
    the selection persists across Streamlit reruns via session_state.
    """
    responses: dict = message.get("responses", {})
    sources = message.get("sources", [])
    picked = message.get("picked")
    question = message.get("question", "")
    turn_id = message.get("turn_id", str(message_index))

    model_keys = list(responses.keys())
    if not model_keys:
        st.warning("No dual responses to display.")
        return

    cols = st.columns(len(model_keys))
    for col, model_key in zip(cols, model_keys):
        label = _LLM_LABELS.get(model_key, model_key)
        resp = responses[model_key]
        is_winner = picked == model_key
        is_loser = picked is not None and not is_winner

        with col:
            winner_mark = " 🏆" if is_winner else ""
            conf_badge = render_confidence_badge(resp.get("confidence"))
            st.markdown(
                f"<div style='margin-bottom: 0.4rem;'>"
                f"<strong>🤖 {label}{winner_mark}</strong>{conf_badge}"
                f"</div>",
                unsafe_allow_html=True,
            )

            scored = resp.get("grounded_sentences", [])
            highlighted = render_with_highlights(scored) if scored else resp.get("answer", "")
            opacity_style = "opacity: 0.55;" if is_loser else ""
            st.markdown(
                f'<div style="{opacity_style}">{highlighted}</div>',
                unsafe_allow_html=True,
            )

            if picked is None:
                if st.button(
                    f"Pick {label}",
                    key=f"pick_{turn_id}_{model_key}",
                    use_container_width=True,
                ):
                    message["picked"] = model_key
                    # Lock subsequent turns to this model for the session.
                    st.session_state.picked_model = model_key
                    record_preference(
                        winner=model_key,
                        question=question,
                        video_id=st.session_state.get("current_video_id"),
                    )
                    st.rerun()

    # Shared sources (shown once beneath both columns)
    if sources:
        render_citations(sources)


def format_overview_message(overview: dict) -> str:
    """
    Render an overview dict as the markdown body of the first assistant
    message (shown before the user can ask anything).
    """
    summary = overview.get("summary", "").strip()
    topics = overview.get("topics", []) or []
    questions = overview.get("suggested_questions", []) or []

    parts: list[str] = ["### 📋 Video Overview"]
    if summary:
        parts.extend(["", summary])

    if topics:
        parts.extend(["", "#### 🏷️ Key Topics", ""])
        parts.extend(f"- {t}" for t in topics)

    if questions:
        parts.extend(["", "#### 💡 Suggested Questions", ""])
        parts.extend(f"- {q}" for q in questions)

    parts.extend(["", "---", "_Feel free to ask any question about the video._"])
    return "\n".join(parts)


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
    grounded_sentences: list[dict] | None = None,
    confidence: int | None = None,
):
    """
    Add an assistant message to session state.

    If grounded_sentences is provided, render_chat_history will re-render
    the answer with per-sentence hallucination highlights on every rerun;
    otherwise the raw content is shown as plain markdown. A non-None
    confidence renders a colored 1-5 badge above the answer.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
        "grounded_sentences": grounded_sentences or [],
        "confidence": confidence,
    })


def add_dual_assistant_message(
    question: str,
    responses: dict,
    sources: list[dict],
    turn_id: str,
):
    """Add a dual-model assistant message to session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "dual": True,
        "question": question,
        "responses": responses,
        "sources": sources,
        "picked": None,
        "turn_id": turn_id,
    })


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
