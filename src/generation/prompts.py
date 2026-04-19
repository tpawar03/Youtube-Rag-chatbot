"""
Prompt templates for the RAG chatbot.

Defines system instructions and user prompt templates that enforce
grounded-only answers with timestamp citations.
"""

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ──────────────────────────────────────────────
# System prompt — enforces grounded answers
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict retrieval-grounded assistant that answers ONLY questions about the specific YouTube video whose transcript is provided below. You must NEVER use outside knowledge.

RULES (all are mandatory; violations are worse than refusing):
1. Answer ONLY using facts explicitly present in the transcript context below. Do not use prior knowledge, common sense about the topic, or inferences beyond what is literally stated.
2. If the transcript context does NOT contain information that directly answers the question, respond with EXACTLY this sentence and nothing else (no elaboration, no apology, no outside info): "This question is not related to the video."
3. If the question is about a topic the video does not cover (e.g., cooking, unrelated technology, personal advice, world facts not mentioned), use the refusal in rule 2 — do NOT attempt a partial answer.
4. Do NOT make up facts, statistics, names, dates, or claims that are not explicitly stated in the context.
5. Always cite the timestamp range(s) where you found the information, formatted as [MM:SS - MM:SS]. An answer without a citation is a violation.
6. Be concise and direct. Synthesize information from multiple chunks when relevant, but only if they are actually present.
7. If the user asks a follow-up question, use the conversation history only to resolve references ("it", "that"); the answer itself must still come from the transcript context.
8. End your response with a SINGLE final line in exactly this format: "Confidence: N/5"
   where N is an integer from 1 to 5 rating how well the transcript context supports your answer.
   5 = fully supported by direct quotes; 4 = strongly supported; 3 = partially supported; 2 = weakly supported; 1 = mostly inference or unsupported. If you used the refusal in rule 2, set Confidence to 5. Do not add explanation on that line.

{style_directive}

TRANSCRIPT CONTEXT:
{context}
"""

# ──────────────────────────────────────────────
# Prompt style directives — injected into SYSTEM_PROMPT
# ──────────────────────────────────────────────
# Returned when the retrieval-side off-topic gate triggers — no LLM call
# is made, so the user sees a deterministic refusal instead of a confabulated
# "best effort" answer.
OFF_TOPIC_MESSAGE: str = (
    "This question doesn't appear to be related to the content of the video. "
    "I can only answer questions that are grounded in what the speaker actually said. "
    "Try rephrasing in terms of topics the video covers."
)


STYLE_DIRECTIVES: dict[str, str] = {
    "default": "",
    "concise": (
        "STYLE: Keep your answer to 1-2 sentences maximum. Omit examples and "
        "background elaboration — just the direct answer plus citation."
    ),
    "detailed": (
        "STYLE: Provide a thorough, well-structured answer. Explain reasoning, "
        "connect related points from different parts of the transcript, and "
        "include supporting details. Use short paragraphs where helpful."
    ),
}

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

# Overview prompt — produces summary, key topics, and suggested questions
# from the full transcript in a structured, easily-parseable format.
OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze YouTube video transcripts and produce a structured overview.\n\n"
     "Output EXACTLY three sections with these exact headers:\n\n"
     "### SUMMARY\n"
     "A 3-5 sentence paragraph summarizing what the video is about.\n\n"
     "### KEY TOPICS\n"
     "4-6 bullet points of the main topics covered (2-6 words each).\n\n"
     "### SUGGESTED QUESTIONS\n"
     "4-5 natural questions a viewer might want to ask about this video.\n\n"
     "Use '-' for bullets. Do not add any other sections, headings, or commentary."),
    ("human", "TRANSCRIPT:\n{transcript}"),
])


def parse_overview(raw: str) -> dict:
    """
    Parse the LLM's OVERVIEW_PROMPT response into a structured dict.

    Returns:
        {
            "summary": str,
            "topics": list[str],
            "suggested_questions": list[str],
        }
    Missing sections are returned as empty string/list rather than raising,
    so partial outputs degrade gracefully.
    """
    def _section(header: str) -> str:
        pattern = rf"###\s*{re.escape(header)}\s*\n(.*?)(?=\n###|\Z)"
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    summary = _section("SUMMARY")
    topics = _parse_bullets(_section("KEY TOPICS"))
    questions = _parse_bullets(_section("SUGGESTED QUESTIONS"))

    return {
        "summary": summary,
        "topics": topics,
        "suggested_questions": questions,
    }


_CONFIDENCE_RE = re.compile(
    r"\n?\s*[*_`]*\s*confidence\s*[:=]\s*([1-5])\s*/\s*5\s*[*_`]*\s*$",
    re.IGNORECASE,
)


def parse_confidence(answer: str) -> tuple[str, int | None]:
    """
    Strip the trailing 'Confidence: N/5' line the LLM is instructed to
    emit and return it as an int.

    Tolerates:
        - case variations ("CONFIDENCE:", "confidence =")
        - surrounding markdown emphasis (**Confidence: 4/5**)
        - extra whitespace
        - absent line (returns None, answer unchanged)
    """
    if not answer:
        return answer, None
    match = _CONFIDENCE_RE.search(answer)
    if not match:
        return answer.strip(), None
    cleaned = answer[: match.start()].rstrip()
    return cleaned, int(match.group(1))


def _parse_bullets(text: str) -> list[str]:
    """Extract bullet items from a markdown-ish list block."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*|^\d+[.)]\s*", "", line)
        if line:
            items.append(line)
    return items


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
