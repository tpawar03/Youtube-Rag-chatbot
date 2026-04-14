"""
Fixed-size token chunking using LangChain's RecursiveCharacterTextSplitter
with a tiktoken tokenizer.

Each chunk carries metadata: video_id, start_time, end_time, chunk_index.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ChunkingConfig


def _build_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Create a token-based text splitter from config."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def fixed_chunk(
    segments: list[dict],
    video_id: str,
    config: ChunkingConfig,
) -> list[dict]:
    """
    Split preprocessed transcript segments into fixed-size token chunks.

    Args:
        segments: Preprocessed segments with text, start, end.
        video_id: YouTube video ID for metadata.
        config: ChunkingConfig with chunk_size and overlap.

    Returns:
        List of chunk dicts with keys:
            - text (str): Chunk text.
            - video_id (str): Source video.
            - start_time (float): Approximate start time.
            - end_time (float): Approximate end time.
            - chunk_index (int): Sequential index.
    """
    # Concatenate all segment texts, but maintain a mapping
    # from character offset → segment index for timestamp recovery
    full_text = ""
    char_to_segment = []  # (char_start, char_end, segment_index)

    for i, seg in enumerate(segments):
        char_start = len(full_text)
        full_text += seg["text"] + " "
        char_end = len(full_text)
        char_to_segment.append((char_start, char_end, i))

    # Split into chunks
    splitter = _build_splitter(config)
    text_chunks = splitter.split_text(full_text)

    # Map each chunk back to timestamps
    chunks = []
    search_start = 0

    for idx, chunk_text in enumerate(text_chunks):
        # Find where this chunk appears in the full text
        pos = full_text.find(chunk_text.strip()[:50], search_start)
        if pos == -1:
            pos = search_start

        chunk_end_pos = pos + len(chunk_text)

        # Find the segments that overlap with this chunk
        start_time = segments[0]["start"]
        end_time = segments[-1]["end"]

        for char_start, char_end, seg_idx in char_to_segment:
            if char_start <= pos < char_end:
                start_time = segments[seg_idx]["start"]
            if char_start < chunk_end_pos <= char_end:
                end_time = segments[seg_idx]["end"]
                break

        chunks.append({
            "text": chunk_text.strip(),
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "chunk_index": idx,
        })

        search_start = max(search_start, pos + 1)

    return chunks
