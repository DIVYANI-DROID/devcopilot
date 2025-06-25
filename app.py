import streamlit as st
from rag.retriever import get_answer

st.set_page_config(page_title="DevCopilot - AI Assistant", layout="wide")

st.title("🧠DevCopilot")
st.caption("Ask questions based on your own notes using a local RAG pipeline.")

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        answer = get_answer(user_input)

        # Store in session state
        st.session_state.chat_history.append((user_input, answer))

# Display history
for question, response in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {question}")
    st.markdown(f"**DevCopilot:** {response}")
    st.markdown("---")
