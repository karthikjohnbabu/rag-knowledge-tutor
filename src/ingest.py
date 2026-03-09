from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

DOC_PATH = "docs/sql_notes.pdf"
DB_PATH = "vectorstore"

loader = PyPDFLoader(DOC_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

vectorstore.persist()

print("Documents indexed successfully.")