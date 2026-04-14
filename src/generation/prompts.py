"""
Prompt templates for the RAG chatbot.

Defines system instructions and user prompt templates that enforce
grounded-only answers with timestamp citations.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ──────────────────────────────────────────────
# System prompt — enforces grounded answers
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about YouTube video content.

RULES:
1. You must answer ONLY based on the provided transcript context below.
2. If the context does not contain enough information to answer the question, say: "I don't have enough information from the video to answer that."
3. Do NOT make up facts, statistics, or claims that are not explicitly stated in the context.
4. Always cite the timestamp range(s) where you found the information, formatted as [MM:SS - MM:SS].
5. Be concise and direct. Synthesize information from multiple chunks when relevant.
6. If the user asks a follow-up question, use the conversation history to understand what they're referring to.

TRANSCRIPT CONTEXT:
{context}
"""

# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

# For the conversational RAG chain
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# For standalone question generation (condenses follow-up into standalone)
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the following conversation and a follow-up question, "
     "rephrase the follow-up question to be a standalone question "
     "that captures the full intent. Do NOT answer the question, "
     "just rephrase it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Simple single-turn QA prompt (for evaluation)
SIMPLE_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string with timestamps.

    Args:
        chunks: List of chunk dicts with text, start_time, end_time.

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        start = _seconds_to_mmss(chunk.get("start_time", 0))
        end = _seconds_to_mmss(chunk.get("end_time", 0))
        context_parts.append(
            f"[Chunk {i} | {start} - {end}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
