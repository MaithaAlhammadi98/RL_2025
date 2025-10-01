# src/backend/state.py
from dataclasses import dataclass
@dataclass(frozen=True)
class EncodedState:
    # compact, hashable representation
    topic: str
    company: str|None

def encode_state(prompt: str, company_info: dict|None) -> EncodedState:
    topic = "legal" if "sue" in prompt.lower() else "general"
    comp = company_info.get("name") if company_info else None
    return EncodedState(topic=topic, company=comp)
