"""
Prompt templates for the RAG chatbot.

Defines system instructions and user prompt templates that enforce
grounded-only answers with timestamp citations.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ──────────────────────────────────────────────
# Confidence instruction appended to every style
# ──────────────────────────────────────────────
_CONFIDENCE_INSTRUCTION = (
    "\n\nAfter your answer, on a NEW LINE write exactly this (replace X with a number 1-5):\n"
    "CONFIDENCE: X/5\n"
    "where 1 = very uncertain, 5 = very confident the context fully supports your answer."
)

# ──────────────────────────────────────────────
# System prompts — one per style
# ──────────────────────────────────────────────
_BASE_RULES = (
    "1. Answer ONLY based on the provided transcript context.\n"
    "2. If the context lacks enough information say: \"I don't have enough information from the video to answer that.\"\n"
    "3. Do NOT make up facts not explicitly stated in the context.\n"
    "4. Cite timestamp ranges where you found the information as [MM:SS - MM:SS].\n"
    "5. Use conversation history to understand follow-up questions."
)

SYSTEM_PROMPTS = {
    "default": (
        "You are a helpful assistant that answers questions about YouTube video content.\n\n"
        "RULES:\n" + _BASE_RULES + "\n6. Be concise and direct.\n\n"
        "TRANSCRIPT CONTEXT:\n{context}" + _CONFIDENCE_INSTRUCTION
    ),
    "concise": (
        "You are a concise assistant answering questions about a YouTube video. "
        "Keep answers to 2-3 sentences maximum. Cut all filler — only state facts.\n\n"
        "RULES:\n" + _BASE_RULES + "\n6. Never exceed 3 sentences.\n\n"
        "TRANSCRIPT CONTEXT:\n{context}" + _CONFIDENCE_INSTRUCTION
    ),
    "detailed": (
        "You are a thorough assistant answering questions about a YouTube video. "
        "Provide comprehensive, well-structured answers with full context and examples from the transcript.\n\n"
        "RULES:\n" + _BASE_RULES + "\n6. Use bullet points or numbered lists for multi-part answers.\n"
        "7. Explain the *why* and *how* behind each point.\n\n"
        "TRANSCRIPT CONTEXT:\n{context}" + _CONFIDENCE_INSTRUCTION
    ),
    "eli5": (
        "You are a friendly assistant explaining YouTube video content as if talking to a curious 10-year-old. "
        "Use simple words, fun analogies, and short sentences. Avoid jargon.\n\n"
        "RULES:\n" + _BASE_RULES + "\n6. Use analogies and real-world comparisons.\n"
        "7. Keep sentences short and vocabulary simple.\n\n"
        "TRANSCRIPT CONTEXT:\n{context}" + _CONFIDENCE_INSTRUCTION
    ),
}

PROMPT_STYLE_LABELS = {
    "default": "Default",
    "concise": "Concise",
    "detailed": "Detailed",
    "eli5": "ELI5 (Simple)",
}

# ──────────────────────────────────────────────
# Prompt template builder
# ──────────────────────────────────────────────

def get_conversational_prompt(style: str = "default") -> ChatPromptTemplate:
    """Return the ChatPromptTemplate for the given style."""
    system = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["default"])
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])


# Default prompt (kept for backward compatibility)
SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]
CONVERSATIONAL_PROMPT = get_conversational_prompt("default")

# ──────────────────────────────────────────────
# Other prompts
# ──────────────────────────────────────────────

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the following conversation and a follow-up question, "
     "rephrase the follow-up question to be a standalone question "
     "that captures the full intent. Do NOT answer the question, "
     "just rephrase it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

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
