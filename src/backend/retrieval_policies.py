"""
Maps an RL action -> retrieval metadata filter for your RAG.

Adjust the field names to match your Chroma/metadata.
By default we assume your chunk metadata contains:
  - "doc_type": e.g. "legal", "financial", "company", ...
  - "company":  a company name (optional)
"""

from __future__ import annotations
from typing import Dict, Any, Optional

ACTIONS = [
    "broad",
    "legal_only",
    "financial_only",
    "company_only",
]

def action_to_filter(action: str, company_name: Optional[str] = None) -> Dict[str, Any] | None:
    """
    Returns a filter dict you can pass into your RAG layer.
    If your RAG call expects a different shape, modify here only.
    """
    a = (action or "").lower()

    if a == "broad":
        return None  # no constraints

    if a == "legal_only":
        return {"doc_type": "legal"}

    if a == "financial_only":
        return {"doc_type": "financial"}

    if a == "company_only":
        # Prefer company-specific docs. If you don't have per-company docs,
        # change this to whatever key your metadata uses (e.g., "source": "company_guidelines.pdf")
        if company_name:
            return {"company": company_name}
        # fallback to a generic company materials flag
        return {"doc_type": "company"}

    # unknown action -> no filter
    return None
