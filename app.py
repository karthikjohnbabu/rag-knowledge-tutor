import streamlit as st
from src.rag_chain import ask_question

st.title("Zentric SQL AI Tutor")

question = st.text_input("Ask a SQL Question")

if question:
    result = ask_question(question)

    st.write("### Answer")
    st.write(result["result"])

    st.write("### Sources")

    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "SQL Notes")

        st.write(f"📄 {source} — Page {page}")

if st.checkbox("Show Retrieved Chunks"):
    for doc in result["source_documents"]:
        st.write(doc.page_content)