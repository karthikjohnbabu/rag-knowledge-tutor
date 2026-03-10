from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def rewrite_query(question, chat_history):

    if not chat_history:
        return question

    history_text = "\n".join([f"User: {c['question']}" for c in chat_history[-3:]])

    prompt = f"""
You are a query rewriting assistant.

Rewrite the user's latest question so that it becomes a fully standalone question.

Conversation:
{history_text}

User question:
{question}

Rewrite the question clearly.
"""

    response = llm.invoke(prompt)

    return response.content.strip()