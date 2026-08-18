import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

 
class Config:
 
    def __init__(self):

        self.chat_model = "gemini-3.5-flash"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.temperature = 0.0        
 

        self.docs_folder = "./uploaded_docs"
 
        self.chunk_size = 1000
        self.chunk_overlap = 150
 
        self.top_k = 6
        self.max_history_turns = 10
 
    
        self.api_key = os.getenv("GEMINI_API_KEY", "")
 
    def validate(self):
        if not self.api_key:
            raise SystemExit(
                "No API key found. Set GEMINI_API_KEY first, e.g.:\n"
                "  export GEMINI_API_KEY='your-key'   (macOS/Linux)\n"
                '  setx GEMINI_API_KEY "your-key"     (Windows)'
            )