"""
Turns a (prompt, company_info) into a coarse-grained state key that is
stable and small enough for a Q-table.

You can change the buckets/keywords at any time without touching the RL agent.
"""

from __future__ import annotations
from typing import Dict, Any
import json
import re
from datetime import datetime


TOPIC_BUCKETS = {
    "legal":  [r"law|legal|sue|regulat|policy|compliance|malpractice"],
    "fin":    [r"budget|cost|price|fund|profit|finance|investment"],
    "ghg":    [r"ghg|emission|carbon|co2|footprint|offset|net\s*zero|scope\s*[123]"],
    "other":  [r".*"],  # catch-all
}


def _bucket(text: str, buckets: Dict[str, list[str]]) -> str:
    text = text.lower()
    for name, patterns in buckets.items():
        for p in patterns:
            if re.search(p, text):
                return name
    return "other"


def encode_state(prompt: str, company_info: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Returns a small dict; the RL agent will stringify this to use as a key.
    """
    topic = _bucket(prompt, TOPIC_BUCKETS)
    length_bucket = "short" if len(prompt) < 80 else "medium" if len(prompt) < 200 else "long"
    sector = (company_info or {}).get("sector", "unknown").lower()
    size   = (company_info or {}).get("size",   "unknown").lower()

    # Include a very light notion of recency to prevent overfitting to exact dialogs
    month = datetime.utcnow().strftime("%Y-%m")

    state = {
        "topic": topic,
        "len": length_bucket,
        "sector": sector,
        "size": size,
        "month": month,
    }
    return state


def state_key(state: Dict[str, Any]) -> str:
    """Stable string key for dict state (sorted keys -> deterministic)."""
    return json.dumps(state, sort_keys=True)
