import streamlit as st
import os

from config import Config
from loaders import load_folder, SUPPORTED_EXTENSION
from vector_store import VectorStoreManager
from chatbot import RAGChatbot

st.set_page_config(page_title="RAG Research Assistant ChatBot:")
st.title(" Research Assistant ChatBot")
st.caption("Ask questions about your documents. Answers " \
"come only from what you upload")

def build_chatbot(docs_folder: str) ->RAGChatbot:
    config = Config()
    config.docs_folder = docs_folder
    config.validate()
    documents = load_folder(docs_folder)
    store = VectorStoreManager(config)
    store.build(documents)
    return RAGChatbot(config, store)

with st.sidebar:
    st.header("1. Add documents")
    st.write("Supported: " + ", ".join(SUPPORTED_EXTENSION))
    uploaded = st.file_uploader(
        "Upload Files", 
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSION],
        accept_multiple_files=True
    )

    if st.button("Build knowledge base", type="primary"):
        if not uploaded:
            st.warning("Upload at least one file first.")
        else:
            docs_folder = "./uploaded_docs"
            os.makedirs(docs_folder, exist_ok=True)

            for old in os.listdir(docs_folder):
                os.remove(os.path.join(docs_folder, old))

    
            for f in uploaded:
                with open(os.path.join(docs_folder, f.name), "wb") as out:
                    out.write(f.getbuffer())
            with st.spinner("Embedding and indexing..."):
                st.session_state.chatbot = build_chatbot(docs_folder)
                st.session_state.messages = []
            st.success(f"Indexed {len(uploaded)} file(s). Ask away!")

    st.header("2. Chat")
    if st.button("Clear chat"):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

# replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write("- " + s)

question = st.chat_input("Ask a question about your documents...")

if question:
    if "chatbot" not in st.session_state:
        st.warning("Build the knowledge base first (sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.chatbot.ask(question)
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write("- " + s)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources })
