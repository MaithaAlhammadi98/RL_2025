"""
Simple reward helpers from UI feedback.
Use feedback_reward('up') or feedback_reward('down').
"""

from __future__ import annotations

def feedback_reward(tag: str) -> float:
    tag = (tag or "").lower()
    if tag in ("up", "thumbs_up", "good", "helpful", "👍"):
        return 1.0
    if tag in ("down", "thumbs_down", "bad", "not_helpful", "👎"):
        return -1.0
    return 0.0  # neutral/unknown

def scale_reward(score: float, min_r: float = -1.0, max_r: float = 1.0) -> float:
    """Clamp external scores into [-1, 1] (or custom bounds)."""
    score = float(score)
    score = min(max(score, min_r), max_r)
    return score
