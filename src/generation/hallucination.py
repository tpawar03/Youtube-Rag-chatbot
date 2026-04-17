"""
Hallucination detector using token-overlap grounding scores.

Splits the answer into sentences and scores each one against the
retrieved chunks. Low-overlap sentences are flagged as potentially
ungrounded (hallucinated).
"""

import re


_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "but", "for", "with", "that", "this", "was", "are", "be",
    "as", "by", "from", "i", "he", "she", "they", "we", "you",
}

# Sentence with at least 5 meaningful words gets a grounding check
_MIN_TOKENS = 5


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _overlap_score(sentence_tokens: set[str], chunk_text: str) -> float:
    """Jaccard-like overlap: |sentence ∩ chunk| / |sentence|."""
    if not sentence_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def score_sentences(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Score each sentence in the answer for grounding in the chunks.

    Args:
        answer: The LLM-generated answer text.
        chunks: Retrieved source chunks (each has a "text" key).

    Returns:
        List of dicts: {sentence, score, grounded}
        score is the max token overlap with any chunk (0.0–1.0).
        grounded is True if score >= 0.25.
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    chunk_texts = [c.get("text", "") for c in chunks]
    results = []

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        tokens = _tokenize(sent)
        if len(tokens) < _MIN_TOKENS:
            # Too short to score meaningfully — treat as grounded
            results.append({"sentence": sent, "score": 1.0, "grounded": True})
            continue
        max_score = max(
            (_overlap_score(tokens, ct) for ct in chunk_texts),
            default=0.0,
        )
        results.append({
            "sentence": sent,
            "score": round(max_score, 3),
            "grounded": max_score >= 0.25,
        })

    return results
