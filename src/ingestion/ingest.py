from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.ingestion.loader import load_documents
from src.utils.config import (
    VECTORSTORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    USE_SEMANTIC_CHUNKING,
)


def ingest_documents():
    documents = load_documents()
    print(f"Loaded {len(documents)} document pages")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if USE_SEMANTIC_CHUNKING:
        print("Using SemanticChunker...")
        text_splitter = SemanticChunker(embeddings)
    else:
        print("Using RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    vectorstore.persist()
    print("Vector database created successfully.")


if __name__ == "__main__":
    ingest_documents()