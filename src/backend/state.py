# src/backend/state.py
from sentence_transformers import SentenceTransformer

CLASSES = ["scope3", "disclosure", "calc", "general"]

def classify_query(q: str) -> str:
    q_low = q.lower()
    if "scope 3" in q_low or "scope3" in q_low: return "scope3"
    if any(k in q_low for k in ["deadline","reporting","disclose","when do","due date"]): return "disclosure"
    if any(k in q_low for k in ["calculate","methodology","formula","how do i calculate"]): return "calc"
    return "general"

def encode_state(query: str, company_profile: dict) -> str:
    kind = classify_query(query)
    size = (company_profile or {}).get("size","medium").lower()
    sector = (company_profile or {}).get("sector","general").lower()
    return f"{kind}|{size}|{sector}"
