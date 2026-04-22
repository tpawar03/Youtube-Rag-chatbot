"""
Sentence-level hallucination detector.

For each sentence in an answer, compute bigram Jaccard overlap with the
concatenated retrieved context. Sentences below the threshold are flagged
as "ungrounded" and can be rendered with a visual marker in the UI.

This is a heuristic — cheap, interpretable, no extra model call. It catches
blatantly fabricated content but will false-flag sentences that paraphrase
heavily. Tune GROUNDING_THRESHOLD if too noisy.
"""

from __future__ import annotations


import re
import string


GROUNDING_THRESHOLD = 0.15  # bigram-Jaccard floor for "grounded"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")


def _tokenize(text: str) -> list[str]:
    return text.lower().translate(_PUNCT_TABLE).split()


def _bigrams(text: str) -> set[tuple[str, str]]:
    toks = _tokenize(text)
    return set(zip(toks, toks[1:]))


def split_sentences(text: str) -> list[str]:
    """
    Split answer text into sentences. Handles the common patterns in LLM
    output; not a full NLP sentence splitter.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def score_sentences(
    answer: str,
    source_chunks: list[dict],
    threshold: float = GROUNDING_THRESHOLD,
) -> list[dict]:
    """
    Score each sentence's grounding against concatenated source chunk text.

    Args:
        answer: The generated answer text.
        source_chunks: Retrieved chunks (each with a "text" field).
        threshold: Jaccard overlap below which a sentence is ungrounded.

    Returns:
        List of {sentence, is_grounded, score} in original order.
    """
    context_bigrams = set()
    for chunk in source_chunks:
        context_bigrams |= _bigrams(chunk.get("text", ""))

    results = []
    for sent in split_sentences(answer):
        sb = _bigrams(sent)
        # Very short sentences (no bigrams) can't be meaningfully scored —
        # mark as grounded so we don't spam false positives on e.g. "Yes.".
        if not sb:
            score = 1.0
        else:
            overlap = len(sb & context_bigrams) / len(sb)
            score = overlap
        results.append({
            "sentence": sent,
            "is_grounded": score >= threshold,
            "score": round(score, 3),
        })
    return results


def answer_is_grounded(answer: str, source_chunks: list[dict]) -> bool:
    """
    Post-generation safety net: returns True if AT LEAST ONE sentence of
    the answer has bigram overlap >= GROUNDING_THRESHOLD with the retrieved
    chunks. A completely ungrounded answer is treated as a sign that the
    LLM ignored the context — the caller should replace it with a refusal.
    """
    scored = score_sentences(answer, source_chunks)
    return any(s["is_grounded"] for s in scored)


def render_with_highlights(scored: list[dict]) -> str:
    """
    Join scored sentences into markdown, wrapping ungrounded ones in a
    white-dashed-underline span. Returned string is safe to pass to
    st.markdown(..., unsafe_allow_html=True).
    """
    parts = []
    for item in scored:
        sent = item["sentence"]
        if item["is_grounded"]:
            parts.append(sent)
        else:
            parts.append(
                f'<span style="border-bottom: 2px dashed #ffffff; '
                f'text-decoration: none; padding-bottom: 1px;" '
                f'title="Possibly ungrounded (score {item["score"]})">'
                f"{sent}</span>"
            )
    return " ".join(parts)
