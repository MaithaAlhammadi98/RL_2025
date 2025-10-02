# inside display_ghg_consultant() after you read `prompt`
from backend.state import encode_state
from backend.rl_agent import RlAgent
from backend.retrieval_policies import action_to_filter
from backend.reward import feedback_reward

if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = RlAgent()  # persists Q in data/q_table.json

state = encode_state(prompt, st.session_state.get("company_info"))
action = st.session_state.rl_agent.select(state)
meta_filter = action_to_filter(action)

# 1) use your existing RAG, but pass `meta_filter` to constrain retrieval
relevant_chunks, metadatas = st.session_state.rag_class.query_documents(
    question=prompt, n_results=4, metadata_filter=meta_filter  # add this arg in rag_process
)

# 2) generate the answer as before
response = st.session_state.ghg_assistant.generate_response(
    user_prompt=prompt,
    relevant_chunks=relevant_chunks,
    results_metadata=metadatas
)

# 3) show answer + feedback buttons
st.markdown(response)
col1, col2 = st.columns(2)
if col1.button("👍 Helpful"):
    r = feedback_reward("up")
    st.session_state.rl_agent.update(state, action, r)
    st.success("Thanks! Learning updated.")
if col2.button("👎 Not helpful"):
    r = feedback_reward("down")
    st.session_state.rl_agent.update(state, action, r)
    st.info("Got it. Learning updated.")
