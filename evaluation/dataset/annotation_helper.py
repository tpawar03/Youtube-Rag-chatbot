"""
Semi-automated QA pair annotation helper.

Given a YouTube video ID:
    1. Fetches and preprocesses the transcript.
    2. Chunks it using the default config.
    3. Uses the LLM to generate candidate questions.
    4. Drafts reference answers and maps to relevant chunks.
    5. Outputs a draft qa_pairs.json for human review.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PipelineConfig, EVAL_DIR
from src.transcript.fetcher import fetch_transcript, extract_video_id
from src.transcript.preprocessor import preprocess_transcript
from src.chunking.fixed_chunker import fixed_chunk
from src.generation.llm import create_llm
from src.generation.prompts import _seconds_to_mmss

logger = logging.getLogger(__name__)


QUESTION_GENERATION_PROMPT = """You are given a chunk of a YouTube video transcript with timestamps.
Generate exactly 3 factual questions that can be answered using ONLY the information in this chunk.

Rules:
1. Questions must be specific and factual (not vague or opinion-based).
2. Questions should require understanding the content, not just keyword matching.
3. Include a mix of: what/who/when/how questions.
4. For each question, provide:
   - The question text
   - A concise reference answer (1-2 sentences)

TRANSCRIPT CHUNK [{chunk_id}] ({start_time} - {end_time}):
{chunk_text}

Respond in this exact JSON format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""


def generate_qa_pairs(
    video_url: str,
    domain: str,
    config: PipelineConfig = None,
    max_chunks: int = 10,
) -> list[dict]:
    """
    Generate candidate QA pairs for a video.

    Args:
        video_url: YouTube URL or video ID.
        domain: Domain label (e.g., "news_analysis", "cs_lectures").
        config: Pipeline config (defaults to PipelineConfig()).
        max_chunks: Maximum number of chunks to generate questions from.

    Returns:
        List of QA pair dicts ready for human review.
    """
    if config is None:
        config = PipelineConfig()

    video_id = extract_video_id(video_url)

    # Fetch and preprocess
    logger.info(f"Fetching transcript for {video_id}...")
    raw_segments = fetch_transcript(video_id)
    segments = preprocess_transcript(raw_segments)

    # Chunk
    chunks = fixed_chunk(segments, video_id, config.chunking)
    logger.info(f"Created {len(chunks)} chunks")

    # Select chunks to generate questions from (spread across video)
    if len(chunks) > max_chunks:
        step = len(chunks) // max_chunks
        selected = [chunks[i] for i in range(0, len(chunks), step)][:max_chunks]
    else:
        selected = chunks

    # Create LLM
    try:
        llm = create_llm(config.generation)
    except Exception as e:
        logger.error(f"Could not create LLM: {e}")
        logger.info("Generating template QA pairs without LLM...")
        return _generate_template_qa(selected, video_id, domain)

    # Generate questions per chunk
    qa_pairs = []

    for chunk in selected:
        prompt = QUESTION_GENERATION_PROMPT.format(
            chunk_id=chunk["chunk_index"],
            start_time=_seconds_to_mmss(chunk["start_time"]),
            end_time=_seconds_to_mmss(chunk["end_time"]),
            chunk_text=chunk["text"][:1500],
        )

        try:
            response = llm.invoke(prompt)

            # Parse JSON from response
            json_match = _extract_json(response)
            if json_match:
                for qa in json_match:
                    qa_pairs.append({
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "relevant_chunk_ids": [chunk["chunk_index"]],
                        "video_id": video_id,
                        "domain": domain,
                        "auto_generated": True,
                        "reviewed": False,
                    })
        except Exception as e:
            logger.warning(f"Failed to generate QA for chunk {chunk['chunk_index']}: {e}")

    logger.info(f"Generated {len(qa_pairs)} candidate QA pairs")
    return qa_pairs


def _extract_json(text: str) -> list[dict] | None:
    """Try to extract a JSON array from LLM output."""
    import re
    # Find JSON array in the response
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _generate_template_qa(
    chunks: list[dict], video_id: str, domain: str
) -> list[dict]:
    """Generate template QA pairs when LLM is unavailable."""
    qa_pairs = []
    for chunk in chunks:
        qa_pairs.append({
            "question": f"[TODO] Question about content at {_seconds_to_mmss(chunk['start_time'])}?",
            "answer": f"[TODO] Answer based on: {chunk['text'][:100]}...",
            "relevant_chunk_ids": [chunk["chunk_index"]],
            "video_id": video_id,
            "domain": domain,
            "auto_generated": False,
            "reviewed": False,
        })
    return qa_pairs


def save_qa_pairs(qa_pairs: list[dict], output_path: Path = None):
    """Save QA pairs to JSON file."""
    if output_path is None:
        output_path = EVAL_DIR / "dataset" / "qa_pairs.json"

    # Merge with existing if file exists
    existing = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)

    combined = existing + qa_pairs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Saved {len(combined)} QA pairs to {output_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate QA pairs for evaluation")
    parser.add_argument("--video-url", required=True, help="YouTube URL or video ID")
    parser.add_argument("--domain", required=True, help="Domain label")
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pairs = generate_qa_pairs(
        video_url=args.video_url,
        domain=args.domain,
        max_chunks=args.max_chunks,
    )

    output = Path(args.output) if args.output else None
    save_qa_pairs(pairs, output)

    print(f"\n✅ Generated {len(pairs)} QA pairs")
    for i, qa in enumerate(pairs[:5], 1):
        print(f"\n  Q{i}: {qa['question']}")
        print(f"  A{i}: {qa['answer'][:100]}...")
