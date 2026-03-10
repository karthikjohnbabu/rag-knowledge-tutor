from sentence_transformers import CrossEncoder


# Load model once
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank(query, documents, top_k=3):

    pairs = [[query, doc.page_content] for doc in documents]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(documents, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = []

    for doc, score in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs