# src/backend/retrieval_policies.py
ACTIONS = ["AU", "US", "EU", "GLOBAL"]  # Tier 1, Tier 2a, Tier 2b, Tier 3

def action_to_filter(action: str):
    # Return a metadata filter your Chroma query understands
    if action == "AU":     return {"tier": "AU"}       # Tier 1
    if action == "US":     return {"tier": "US"}       # Tier 2a
    if action == "EU":     return {"tier": "EU"}       # Tier 2b
    if action == "GLOBAL": return {"tier": "GLOBAL"}   # Tier 3
    return {}
