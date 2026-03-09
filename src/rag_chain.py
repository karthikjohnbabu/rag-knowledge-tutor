from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """
You are a SQL tutor for Zentric Institute.

Use the provided SQL notes as reference.

If the student asks for:
- explanation → explain using the notes
- example → generate example SQL queries
- exercises → generate a practice exercise based on the concept

If the answer is not present in the notes, you may still generate a helpful teaching response.

Context:
{context}

Question:
{question}

Provide a clear teaching answer suitable for a beginner SQL student.
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

DB_PATH = "vectorstore"

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def ask_question(question):
    result = qa_chain({"query": question})
    return result