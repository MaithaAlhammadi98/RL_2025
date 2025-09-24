# #Async Callbacks workaround

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# Import packages
import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os
from pathlib import Path

# Set Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from app.ghg_consultant import display_ghg_consultant
from app.company_form import display_company_form
from app.embedding_gen_db import *
from backend.rag_process import rag_process
from backend.ghg_assistant import GHGAssistant

if "rag_class" not in st.session_state:
    st.session_state.rag_class = rag_process()

if "ghg_assistant" not in st.session_state:
    st.session_state.ghg_assistant = GHGAssistant()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state["company_info"] = None

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Company Form", "GHG Consultant"],
        icons=["file-earmark-text", "chat"],
        default_index=0,
    )

# Display the selected page
if selected == "GHG Consultant":
    display_ghg_consultant()
elif selected == "Company Form":
    display_company_form()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by NLP Group 10")
st.sidebar.markdown("🧑‍💻 Built with Streamlit")
