import streamlit as st
import os

from src.retrieval.rag_chain import ask_question
from src.ingestion.ingest import ingest_documents


# ---------- Session State Initialization ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


DOCS_FOLDER = "docs"

st.title("RAG Knowledge Tutor")
st.write("Upload learning material and ask questions.")

os.makedirs(DOCS_FOLDER, exist_ok=True)


# ---------- File Upload Section ----------

uploaded_file = st.file_uploader(
    "Upload learning material (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:

    save_path = os.path.join(DOCS_FOLDER, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("Process Document"):
        with st.spinner("Indexing document..."):
            ingest_documents()

        st.success("Document indexed successfully!")


# ---------- Question Section ----------

st.write("## Ask a Question")

question = st.text_input(
    "Ask a question",
    key="question_input"
)

ask_button = st.button("Ask")


if ask_button and question:

    with st.spinner("Thinking..."):

        result = ask_question(question, st.session_state.chat_history)

        # Save conversation
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"],
            "debug": result.get("vector_debug", []),
            "query_type": result.get("query_type", None)
        })


# ---------- Conversation History ----------

if st.session_state.chat_history:

    st.write("## Conversation")

    for chat in reversed(st.session_state.chat_history):

        st.write("### You")
        st.write(chat["question"])

        st.write("### Tutor")
        st.write(chat["answer"])

        if chat["query_type"]:
            st.write(f"**Detected Query Type:** `{chat['query_type']}`")

        st.write("### Sources")

        for doc in chat["sources"]:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "Document")

            st.write(f"📄 {source} — Page {page}")

        # ---------- Debug Tools ----------

        if st.checkbox("Show Retrieved Chunks (Debug Mode)", key=f"chunks_{chat['question']}"):

            st.write("### Retrieved Context")

            for idx, doc in enumerate(chat["sources"], start=1):

                page = doc.metadata.get("page", "N/A")
                source = doc.metadata.get("source", "Document")

                st.write(f"**Chunk {idx}** — {source} — Page {page}")
                st.write(doc.page_content)
                st.write("---")


        if st.checkbox("Show Vector Scores (Advanced Debug)", key=f"debug_{chat['question']}"):

            st.write("### Vector Retrieval Debug")

            for item in chat["debug"]:

                st.write(
                    f"**Rank {item['rank']}** | "
                    f"Score: {item['score']:.4f} | "
                    f"{item['source']} — Page {item['page']}"
                )

                st.write(item["content"][:600])
                st.write("---")