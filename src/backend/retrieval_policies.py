from __future__ import annotations
from typing import Dict, Any, Optional

ACTIONS = ["broad", "legal_only", "financial_only", "company_only"]

def action_to_filter(action: str, company_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    a = (action or "").lower()
    if a == "legal_only":
        return {"doc_type": "legal"}
    if a == "financial_only":
        return {"doc_type": "financial"}
    if a == "company_only" and company_name:
        # use the metadata field name you actually store (e.g. "company", "org", etc.)
        return {"company": company_name}
    # "broad" or unknown → no filter
    return None



