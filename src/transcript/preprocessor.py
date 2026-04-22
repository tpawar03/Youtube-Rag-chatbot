"""
Preprocess raw YouTube transcripts.

Cleans filler words, restores basic punctuation, and merges
short segments into coherent paragraphs while preserving
timestamp mappings for citation.
"""

from __future__ import annotations


import re
from typing import Optional


# Common English filler words/phrases to remove
FILLER_PATTERNS = [
    r"\b(um+)\b",
    r"\b(uh+)\b",
    r"\b(er+)\b",
    r"\b(ah+)\b",
    r"\b(like)\b(?=\s*,?\s*(?:you know|I mean|basically|literally|so|um|uh))",
    r"\b(you know)\b",
    r"\b(I mean)\b",
    r"\b(kind of)\b",
    r"\b(sort of)\b",
    r"\b(basically)\b(?=\s*,?\s*(?:like|you know|I mean|so))",
]

# Compiled for performance
_filler_regex = re.compile(
    "|".join(FILLER_PATTERNS), re.IGNORECASE
)


def remove_fillers(text: str) -> str:
    """Remove common filler words from transcript text."""
    cleaned = _filler_regex.sub("", text)
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Remove orphaned commas
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    return cleaned


def restore_punctuation(text: str) -> str:
    """
    Apply heuristic punctuation restoration to raw caption text.

    This is a lightweight approach — not as accurate as a
    punctuation-restoration model, but sufficient for chunking.
    """
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Add period at end if missing
    if text and text[-1] not in ".!?":
        text = text + "."

    # Capitalize after sentence-ending punctuation
    text = re.sub(
        r"([.!?])\s+([a-z])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )

    return text


def merge_segments(
    segments: list[dict],
    max_gap_seconds: float = 2.0,
    min_segment_length: int = 20,
) -> list[dict]:
    """
    Merge short consecutive transcript segments into longer paragraphs.

    Args:
        segments: Raw transcript segments from youtube-transcript-api.
        max_gap_seconds: Maximum time gap between segments to merge.
        min_segment_length: Minimum character length before a segment
                           is considered standalone.

    Returns:
        List of merged segments with keys:
            - text (str): Merged and cleaned text.
            - start (float): Start time of first constituent segment.
            - end (float): End time of last constituent segment.
    """
    if not segments:
        return []

    merged = []
    current_text = segments[0]["text"]
    current_start = segments[0]["start"]
    current_end = segments[0]["start"] + segments[0]["duration"]

    for seg in segments[1:]:
        seg_start = seg["start"]
        seg_end = seg["start"] + seg["duration"]
        gap = seg_start - current_end

        # Merge if gap is small or current segment is too short
        if gap <= max_gap_seconds or len(current_text) < min_segment_length:
            current_text += " " + seg["text"]
            current_end = seg_end
        else:
            merged.append({
                "text": current_text,
                "start": current_start,
                "end": current_end,
            })
            current_text = seg["text"]
            current_start = seg_start
            current_end = seg_end

    # Don't forget the last segment
    merged.append({
        "text": current_text,
        "start": current_start,
        "end": current_end,
    })

    return merged


def preprocess_transcript(
    segments: list[dict],
    remove_filler_words: bool = True,
    apply_punctuation: bool = True,
    merge: bool = True,
    max_gap_seconds: float = 2.0,
) -> list[dict]:
    """
    Full preprocessing pipeline for raw transcript segments.

    Args:
        segments: Raw segments from youtube-transcript-api.
        remove_filler_words: Whether to strip filler words.
        apply_punctuation: Whether to apply heuristic punctuation.
        merge: Whether to merge short consecutive segments.
        max_gap_seconds: Max gap for merging.

    Returns:
        List of cleaned, merged segments with text, start, and end times.
    """
    # Step 1: Merge short segments
    if merge:
        processed = merge_segments(segments, max_gap_seconds=max_gap_seconds)
    else:
        processed = [
            {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["start"] + seg["duration"],
            }
            for seg in segments
        ]

    # Step 2: Clean each segment
    for seg in processed:
        text = seg["text"]
        if remove_filler_words:
            text = remove_fillers(text)
        if apply_punctuation:
            text = restore_punctuation(text)
        seg["text"] = text

    # Remove empty segments
    processed = [seg for seg in processed if seg["text"].strip()]

    return processed
