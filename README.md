# RAG Research Assistant Chatbot

A document Q&A chatbot built with RAG (Retrieval-Augmented Generation).
Upload your documents, and it answers questions using only their content.

## Stack
- **LLM:** Google Gemini (`gemini-3.5-flash`)
- **Embeddings:** local `all-MiniLM-L6-v2` (HuggingFace)
- **Vector store:** ChromaDB
- **Framework:** LangChain
- **UI:** Streamlit

## Supported files
PDF, DOCX, TXT, Excel (.xlsx / .xls)

## Setup

1. Clone the repo:
```bash
   git clone https://github.com/mherahh/Rag-research-assistant-chatbot.git
   cd Rag-research-assistant-chatbot
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Add your Gemini API key. Create a `.env` file:

4. ## Run
```bash
streamlit run app.py
```

## How to use
1. Upload one or more documents in the sidebar.
2. Click **Build knowledge base**.
3. Ask questions in the chat. Answers come only from your documents.
