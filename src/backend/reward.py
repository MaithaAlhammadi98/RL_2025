# src/backend/reward.py
import numpy as np
from sentence_transformers import SentenceTransformer, util

_model = None
def _emb():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    return _model

def similarity_reward(pred: str, gold: str) -> float:
    # cosine sim → +1 if good, 0 else; keep it simple/defensible
    e = _emb()
    s = util.cos_sim(e.encode([pred], convert_to_tensor=True),
                     e.encode([gold], convert_to_tensor=True))[0,0].item()
    if s >= 0.6: return +1.0
    if s <= 0.3: return -1.0
    return 0.0

def feedback_reward(thumb: str) -> float:
    return +1.0 if thumb == "up" else -1.0
