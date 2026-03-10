from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.utils.config import (
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    VECTOR_K,
    BM25_K,
    ENSEMBLE_WEIGHTS,
    USE_HYBRID_RETRIEVAL,
    RETRIEVAL_TYPE,
)


def _load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def _load_all_documents_from_vectorstore():
    vectorstore = _load_vectorstore()
    store_data = vectorstore.get(include=["documents", "metadatas"])

    documents = []
    raw_docs = store_data.get("documents", [])
    raw_meta = store_data.get("metadatas", [])

    for doc_text, meta in zip(raw_docs, raw_meta):
        documents.append(Document(page_content=doc_text, metadata=meta or {}))

    return documents


def get_retriever():
    vectorstore = _load_vectorstore()

    vector_retriever = vectorstore.as_retriever(
        search_type=RETRIEVAL_TYPE,
        search_kwargs={"k": VECTOR_K}
    )

    if not USE_HYBRID_RETRIEVAL:
        return vector_retriever

    documents = _load_all_documents_from_vectorstore()

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = BM25_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=ENSEMBLE_WEIGHTS
    )

    return hybrid_retriever


def get_vector_debug_results(query: str):
    vectorstore = _load_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=VECTOR_K)

    debug_results = []
    for rank, (doc, score) in enumerate(results, start=1):
        debug_results.append({
            "rank": rank,
            "score": score,
            "source": doc.metadata.get("source", "Document"),
            "page": doc.metadata.get("page", "N/A"),
            "content": doc.page_content,
        })

    return debug_results