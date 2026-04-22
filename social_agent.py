from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

DEFAULT_SOCIAL_SIGNAL = {
    "type": "SAFE",
    "confidence": 0.10,
    "source": "social",
    "text": "No corroborating social signal found.",
}

DEFAULT_FIXTURES = [
    {
        "type": "FIRE",
        "confidence": 0.78,
        "source": "social",
        "text": "Fire reported in mall food court.",
        "keywords": ["fire", "smoke", "mall", "food court"],
    },
    {
        "type": "MEDICAL",
        "confidence": 0.72,
        "source": "social",
        "text": "Person collapse reported near main entrance.",
        "keywords": ["fall", "collapse", "medical", "entrance"],
    },
]


def _load_fixture_file() -> list[dict]:
    fixture_path = os.getenv("SOCIAL_SIGNAL_FIXTURE_FILE", "").strip()
    if not fixture_path:
        return DEFAULT_FIXTURES

    path = Path(fixture_path).expanduser()
    if not path.exists():
        return DEFAULT_FIXTURES

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_FIXTURES


def fetch_social_signals(query: str, location: str | None = None) -> dict:
    normalized_query = f"{query} {location or ''}".lower()
    fixtures = _load_fixture_file()

    for fixture in fixtures:
        keywords = fixture.get("keywords", [])
        if any(keyword.lower() in normalized_query for keyword in keywords):
            return {
                "type": fixture.get("type", "SAFE"),
                "confidence": float(fixture.get("confidence", 0.5)),
                "source": fixture.get("source", "social"),
                "text": fixture.get("text", "Possible crisis signal reported."),
            }

    fallback = DEFAULT_SOCIAL_SIGNAL.copy()
    if location:
        fallback["text"] = f"{fallback['text']} Location: {location}."
    return fallback
