from operator import itemgetter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
 
 
 
class ConversationMemory:
 
    def __init__(self, max_turns):
        self.max_turns = max_turns
        self.lines = []
 
    def add(self, question, answer):
        self.lines.append("User: " + question)
        self.lines.append("Assistant: " + answer)
        # keep only the most recent lines (2 lines = 1 turn)
        self.lines = self.lines[-(self.max_turns * 2):]
 
    def as_text(self):
        if not self.lines:
            return "(no history yet)"
        return "\n".join(self.lines)
 
    def clear(self):
        self.lines = []
 
 
class RAGChatbot:
   
 
    PROMPT_TEMPLATE = """You are a research assistant. Your job is to help the user
understand their documents accurately and thoroughly.

Rules:
- Answer using ONLY the context below. Do not use outside knowledge.
- If the answer is not in the context, say "I don't know based on the documents."
- Be precise. When the documents give specific numbers, names, or results, use them exactly.
- If the context contains conflicting or partial information, say so rather than smoothing it over.
- Structure longer answers clearly, and keep the tone factual, not salesy.

 
Conversation history:
{history}
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
    def __init__(self, config, vector_store):
        self.config = config
        self.memory = ConversationMemory(config.max_history_turns)
        self.llm = ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.api_key,
            temperature=config.temperature,
        )
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        self.retriever = vector_store.get_retriever()

        self.chain = self.prompt | self.llm | StrOutputParser()
 
    def _format_docs(self, docs):
        blocks =[]

        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            label = source if page is None else f"{source}, page {page}"
            blocks.append(f"[Source: {label}]\n{d.page_content}")
        return "\n\n".join(blocks)

    def _list_sources(self, docs):
        """Unique, readable source labels for the UI."""
        seen = []
        for d in docs:
            source = os.path.basename(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page")
            label = source if page is None else f"{source}, page {page}"
            if label not in seen:
                seen.append(label)
        return seen

 
    def ask(self, question):
        """Answer a question and return (answer, sources)."""
        docs = self.retriever.invoke(question)      # 1. retrieve
        context = self._format_docs(docs)           # 2. format into context
        answer = self.chain.invoke({                # 3. answer
            "question": question,
            "context": context,
            "history": self.memory.as_text(),
        })
        self.memory.add(question, answer)
        return answer, self._list_sources(docs)