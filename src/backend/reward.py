from __future__ import annotations

def feedback_reward(tag: str) -> float:
    t = (tag or "").lower()
    if t in ("up", "👍", "thumbs_up", "good", "helpful", "positive"):
        return 1.0
    if t in ("down", "👎", "thumbs_down", "bad", "not_helpful", "negative"):
        return -1.0
    return 0.0

def scale_reward(score: float, min_r: float = -1.0, max_r: float = 1.0) -> float:
    score = float(score)
    return max(min(score, max_r), min_r)
