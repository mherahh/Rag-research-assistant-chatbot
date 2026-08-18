from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
 
class VectorStoreManager:
    """Builds and queries the vector index behind a small, simple interface."""
 
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model,)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.store = None   # gets created in build()
 
    def build(self, documents):
      #split documents, embed locally and store in chromadb
        chunks = self.splitter.split_documents(documents)
        print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        client = chromadb.EphemeralClient()

    
        self.store = Chroma.from_documents(chunks, self.embeddings, client=client,
         collection_name="rag_docs")
        print("done building the store")

 
    def get_retriever(self):
        """Return a retriever the chatbot can send questions to."""
        if self.store is None:
            raise RuntimeError("Call build() before get_retriever().")
        return self.store.as_retriever(search_kwargs={"k": self.config.top_k})