"""
Persistent preference tracking for dual-model comparisons.

Records which model (mistral / llama2) the user picks each time they
compare responses side-by-side, and maintains running win counts in
data/preferences.json.
"""

from __future__ import annotations


import json
from datetime import datetime
from pathlib import Path

from config import DATA_DIR


PREFERENCES_PATH: Path = DATA_DIR / "preferences.json"


def load_preferences() -> dict:
    """
    Load the current preference state. Returns a fresh empty scaffold if
    the file doesn't exist yet.
    """
    if not PREFERENCES_PATH.exists():
        return {"scores": {}, "history": []}
    try:
        return json.loads(PREFERENCES_PATH.read_text())
    except json.JSONDecodeError:
        # Corrupt file — don't blow up the UI, start fresh.
        return {"scores": {}, "history": []}


def record_preference(winner: str, question: str, video_id: str | None = None) -> dict:
    """
    Increment the winner's score and append the selection to history.
    Returns the updated preferences dict.
    """
    prefs = load_preferences()
    prefs["scores"][winner] = prefs["scores"].get(winner, 0) + 1
    prefs["history"].append({
        "winner": winner,
        "question": question,
        "video_id": video_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    PREFERENCES_PATH.write_text(json.dumps(prefs, indent=2))
    return prefs


def get_scores() -> dict[str, int]:
    """Return just the win-count mapping (empty dict if none yet)."""
    return load_preferences().get("scores", {})
