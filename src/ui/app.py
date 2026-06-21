"""Script to build the UI."""

# pylint: disable=wrong-import-position
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.chatbot.chatbot import Chatbot
from src.retrieve.pipeline import RAGPipeline
from src.retrieve.reranking import BaselineReranker
from src.retrieve.search import VectorSearch
from src.retrieve.transformation import BaselineQueryTransformer
from src.utils import get_index_vector_db


# Page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

K = 5  # Number of chunks to retrieve


# Cache the chatbot initialization (expensive Pinecone + LLM setup)
@st.cache_resource
def load_chatbot() -> Chatbot:
    """Loads the chatbot and caches it to avoid doing it every turn.

    Returns:
        Chatbot.
    """

    index = get_index_vector_db()
    pipeline = RAGPipeline(
        query_transformer=BaselineQueryTransformer(),
        searcher=VectorSearch(index, k=2 * K),
        reranker=BaselineReranker(k=K),
    )

    return Chatbot(rag_pipeline=pipeline)


# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG Chatbot")

# Render all previous messages from history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved chunks under each assistant response
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.text_area(
                        f"Chunk {i}",
                        chunk,
                        height=100,
                        disabled=True,
                        key=f"hist_{msg_idx}_chunk_{i}",
                    )

# Handle new user input
if prompt := st.chat_input("Ask a question..."):
    chatbot = load_chatbot()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.chat(prompt)
            chunks = chatbot.last_chunks
        st.markdown(response)
        # Show retrieved chunks for this response
        with st.expander("Retrieved chunks"):
            for i, chunk in enumerate(chunks, 1):
                st.text_area(
                    f"Chunk {i}",
                    chunk,
                    height=100,
                    disabled=True,
                    key=f"new_chunk_{i}",
                )
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "chunks": chunks}
    )

    st.rerun()

# Clear chat button (only shown when there are messages)
if st.session_state.messages and st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
