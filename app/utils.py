"""Utility helpers for the Streamlit app."""

import json
from datetime import datetime
from pathlib import Path

PREFS_FILE = Path(__file__).parent.parent / "data" / "preferences.json"


def log_preference(winner: str, loser: str, question: str, video_id: str):
    """Append one preference record to data/preferences.json."""
    PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if PREFS_FILE.exists():
        try:
            records = json.loads(PREFS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            records = []
    records.append({
        "timestamp": datetime.now().isoformat(),
        "winner": winner,
        "loser": loser,
        "question": question,
        "video_id": video_id,
    })
    PREFS_FILE.write_text(json.dumps(records, indent=2))


def load_scoreboard() -> dict[str, int]:
    """Return {model_key: win_count} from preferences.json."""
    if not PREFS_FILE.exists():
        return {}
    try:
        records = json.loads(PREFS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    scores: dict[str, int] = {}
    for r in records:
        w = r.get("winner", "")
        if w:
            scores[w] = scores.get(w, 0) + 1
    return scores
