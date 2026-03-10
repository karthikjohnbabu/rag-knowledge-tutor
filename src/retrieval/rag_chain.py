from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from src.retrieval.retriever import get_retriever, get_vector_debug_results
from src.utils.config import CHAT_MODEL, DEBUG
from src.utils.query_classifier import classify_query
from src.retrieval.reranker import rerank
from langchain.memory import ConversationBufferMemory
from src.retrieval.query_rewriter import rewrite_query

from src.prompts.prompt_templates import (
    GENERAL_TUTOR_PROMPT,
    EXPLANATION_PROMPT,
    EXAMPLE_PROMPT,
    EXERCISE_PROMPT,
    INTERVIEW_PROMPT
)


retriever = get_retriever()

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0
)


def select_prompt(question: str):
    query_type = classify_query(question)

    if query_type == "exercise":
        return EXERCISE_PROMPT, query_type
    if query_type == "example":
        return EXAMPLE_PROMPT, query_type
    if query_type == "interview":
        return INTERVIEW_PROMPT, query_type
    if query_type == "explanation":
        return EXPLANATION_PROMPT, query_type

    return GENERAL_TUTOR_PROMPT, query_type


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"
)

def ask_question(question, chat_history=None):

    prompt, query_type = select_prompt(question)

    # -------- Query Rewriting --------
    rewritten_query = rewrite_query(question, chat_history)

    # Step 1: Retrieve chunks
    retrieved_docs = retriever.get_relevant_documents(rewritten_query)

    # Step 2: Rerank chunks
    reranked_docs = rerank(rewritten_query, retrieved_docs)

    # Step 3: Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        chain_type_kwargs={"prompt": prompt}
    )

    # Step 4: Run chain
    result = qa_chain({"query": rewritten_query})

    # Replace sources with reranked docs
    result["source_documents"] = reranked_docs

    if DEBUG:
        result["query_type"] = query_type
        result["rewritten_query"] = rewritten_query

    return result