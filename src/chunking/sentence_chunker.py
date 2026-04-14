"""
Sentence-boundary chunking.

Groups sentences into chunks that respect sentence boundaries,
ensuring no chunk splits mid-sentence. Uses NLTK's sent_tokenize
under the hood.
"""

import nltk
import tiktoken

from config import ChunkingConfig

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def sentence_chunk(
    segments: list[dict],
    video_id: str,
    config: ChunkingConfig,
) -> list[dict]:
    """
    Split preprocessed transcript into chunks along sentence boundaries.

    Groups sentences until the accumulated token count reaches chunk_size,
    then starts a new chunk. Overlap is achieved by carrying the last
    N sentences into the next chunk.

    Args:
        segments: Preprocessed segments with text, start, end.
        video_id: YouTube video ID for metadata.
        config: ChunkingConfig (chunk_size used as target token count).

    Returns:
        List of chunk dicts with keys:
            - text, video_id, start_time, end_time, chunk_index
    """
    # Build a flat list of (sentence, start_time, end_time)
    sentence_records = []

    for seg in segments:
        sentences = nltk.sent_tokenize(seg["text"])
        n_sentences = len(sentences)
        seg_duration = seg["end"] - seg["start"]

        for i, sent in enumerate(sentences):
            # Approximate timestamp per sentence within the segment
            frac_start = i / max(n_sentences, 1)
            frac_end = (i + 1) / max(n_sentences, 1)
            sentence_records.append({
                "text": sent.strip(),
                "start": seg["start"] + frac_start * seg_duration,
                "end": seg["start"] + frac_end * seg_duration,
            })

    if not sentence_records:
        return []

    # Group sentences into chunks
    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_start_time = sentence_records[0]["start"]

    overlap_tokens = config.chunk_overlap
    # How many trailing sentences to carry over for overlap
    overlap_sentences = []

    for rec in sentence_records:
        sent_tokens = _count_tokens(rec["text"])

        if current_tokens + sent_tokens > config.chunk_size and current_sentences:
            # Flush current chunk
            chunk_text = " ".join(s["text"] for s in current_sentences)
            chunks.append({
                "text": chunk_text,
                "video_id": video_id,
                "start_time": chunk_start_time,
                "end_time": current_sentences[-1]["end"],
                "chunk_index": len(chunks),
            })

            # Compute overlap: carry trailing sentences
            overlap_sentences = []
            overlap_tok_count = 0
            for s in reversed(current_sentences):
                s_tok = _count_tokens(s["text"])
                if overlap_tok_count + s_tok > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_tok_count += s_tok

            current_sentences = list(overlap_sentences)
            current_tokens = overlap_tok_count
            chunk_start_time = (
                current_sentences[0]["start"] if current_sentences
                else rec["start"]
            )

        current_sentences.append(rec)
        current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(s["text"] for s in current_sentences)
        chunks.append({
            "text": chunk_text,
            "video_id": video_id,
            "start_time": chunk_start_time,
            "end_time": current_sentences[-1]["end"],
            "chunk_index": len(chunks),
        })

    return chunks
