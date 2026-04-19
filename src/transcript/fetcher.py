"""
Fetch timestamped captions from YouTube videos.

Uses youtube-transcript-api to retrieve captions without requiring
a YouTube Data API key. Handles URL parsing, language fallback,
and common error cases.
"""

import json
import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

from config import TRANSCRIPTS_DIR


def _build_api() -> YouTubeTranscriptApi:
    """
    Construct a YouTubeTranscriptApi. Routes through Webshare rotating
    proxies when WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD are
    set, otherwise uses direct requests.
    """
    user = os.getenv("WEBSHARE_PROXY_USERNAME")
    pw = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if user and pw:
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=user,
                proxy_password=pw,
            )
        )
    return YouTubeTranscriptApi()


class TranscriptFetchError(Exception):
    """Raised when transcript cannot be fetched."""
    pass


def extract_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from a URL or return as-is if already an ID.

    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - Plain VIDEO_ID string
    """
    url_or_id = url_or_id.strip()

    # Already a bare video ID (11 chars, alphanumeric + - + _)
    if re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)

    # youtu.be/VIDEO_ID
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=VIDEO_ID
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        # youtube.com/embed/VIDEO_ID
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[1].split("?")[0]

    raise TranscriptFetchError(
        f"Could not extract video ID from: {url_or_id}"
    )


def fetch_transcript(
    url_or_id: str,
    languages: Optional[list[str]] = None,
) -> list[dict]:
    """
    Fetch the transcript for a YouTube video.

    Args:
        url_or_id: YouTube URL or video ID.
        languages: Ordered list of preferred language codes.
                   Defaults to ["en"].

    Returns:
        List of dicts, each with keys:
            - text (str): The caption text.
            - start (float): Start time in seconds.
            - duration (float): Duration in seconds.

    Raises:
        TranscriptFetchError: If the transcript cannot be retrieved.
    """
    if languages is None:
        languages = ["en"]

    video_id = extract_video_id(url_or_id)

    cache_path = TRANSCRIPTS_DIR / f"{video_id}_raw.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    try:
        api = _build_api()
        fetched = api.fetch(video_id, languages=languages)
        # Convert FetchedTranscriptSnippet objects to plain dicts
        transcript = [
            {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
            for snippet in fetched
        ]
        return transcript

    except TranscriptsDisabled:
        raise TranscriptFetchError(
            f"Transcripts are disabled for video: {video_id}"
        )
    except NoTranscriptFound:
        raise TranscriptFetchError(
            f"No transcript found for video: {video_id} "
            f"in languages: {languages}"
        )
    except VideoUnavailable:
        raise TranscriptFetchError(
            f"Video is unavailable: {video_id}"
        )
    except (IpBlocked, RequestBlocked, requests.exceptions.RequestException) as e:
        raise TranscriptFetchError(
            f"Transcript fetch blocked for {video_id} ({type(e).__name__}). "
            f"The Webshare proxy pool is rate-limited by YouTube — "
            f"try again later or upgrade the Webshare plan."
        )
    except Exception as e:
        raise TranscriptFetchError(
            f"Unexpected error fetching transcript for {video_id}: {e}"
        )
