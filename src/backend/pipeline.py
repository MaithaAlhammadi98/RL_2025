# src/backend/pipeline.py

from typing import Dict, Any, Tuple, List, Optional
from backend.rl_agent import RLAgent
from backend.state import encode_state
from backend.retrieval_policies import action_to_filter
from backend.rag_process import rag_process
from backend.ghg_assistant import GHGAssistant

def answer_with_rl(prompt: str, company: Dict[str, Any], agent: RLAgent) -> Tuple[str, str, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: answer, action, chunks, metas, state_dict
    """
    # 1) state
    s = encode_state(prompt, company)
    # 2) choose action
    a = agent.select(s)
    # 3) map to metadata filter
    meta_filter = action_to_filter(a, company.get("name"))
    # 4) retrieve with filter
    rag = rag_process()
    chunks, metas = rag.query_documents(question=prompt, n_results=4, metadata_filter=meta_filter)
    # 5) generate
    assistant = GHGAssistant()
    answer = assistant.generate_response  # async in some versions—use asyncio.run if needed
    try:
        # if your GHGAssistant.generate_response is async:
        import asyncio
        ans = asyncio.run(assistant.generate_response(prompt, context="\n\n".join(chunks)))
    except TypeError:
        # if it's sync:
        ans = assistant.generate_response(prompt, relevant_chunks=chunks, results_metadata=metas)

    return ans, a, chunks, metas, s
def give_feedback(agent: RLAgent, state: Dict[str, Any], action: str, reward: float) -> None:
    agent.update(state, action, reward)
    agent.save()