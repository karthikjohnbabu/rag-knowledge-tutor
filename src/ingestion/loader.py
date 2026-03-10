from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from src.utils.config import DOCS_PATH
import os


def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):

        file_path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

    return documents