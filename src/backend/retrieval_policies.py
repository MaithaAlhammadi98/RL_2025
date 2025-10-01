from __future__ import annotations
from typing import Dict, Any, Optional

ACTIONS = ["broad", "legal_only", "financial_only", "company_only"]

def action_to_filter(action: str, company_name: Optional[str] = None) -> Dict[str, Any] | None:
    a = (action or "").lower()

    if a == "broad":
        return None
    if a == "legal_only":
        return {"doc_type": "legal"}
    if a == "financial_only":
        return {"doc_type": "financial"}
    if a == "company_only":
        if company_name:
            return {"company": company_name}
        return {"doc_type": "company"}

    return None
