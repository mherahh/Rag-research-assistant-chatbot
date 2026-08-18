import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
 

SUPPORTED_EXTENSION = [".pdf", ".docx", ".txt", ".xlsx", ".xls"]
 
 #parent class
class BaseLoader:
 
    def __init__(self, path):
        self.path = path
 
    def load(self):
        # Child classes must replace this with their own version.
        raise NotImplementedError("Each loader must define its own load().")
 
 
class PDFLoader(BaseLoader):
    def load(self):
        return PyPDFLoader(self.path).load()
 
 
class DocxLoader(BaseLoader):
    def load(self):
        return Docx2txtLoader(self.path).load()
 
 
class TxtLoader(BaseLoader):
    def load(self):
        return TextLoader(self.path, encoding="utf-8").load()
 
 
class ExcelLoader(BaseLoader):
 
    def load(self):
        sheets = pd.read_excel(self.path, sheet_name=None)  # name -> DataFrame
        text = ""
        for name, df in sheets.items():
            text += f"Sheet: {name}\n{df.to_string(index=False)}\n\n"
        return [Document(page_content=text, metadata={"source": self.path})]
 
 
def get_loader(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PDFLoader(path)
    elif ext == ".docx":
        return DocxLoader(path)
    elif ext == ".txt":
        return TxtLoader(path)
    elif ext in (".xlsx", ".xls"):
        return ExcelLoader(path)
    else:
        print("Skipping unsupported file:", path)
        return None
 
 
def load_folder(folder):
    documents = []
    for name in os.listdir(folder):
        loader = get_loader(os.path.join(folder, name))
        if loader is not None:
            documents.extend(loader.load())
    return documents