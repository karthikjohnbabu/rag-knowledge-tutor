from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def grade_sql_answer(question, student_answer):

    prompt = f"""
You are a SQL tutor.

Exercise:
{question}

Student Answer:
{student_answer}

Evaluate the answer and give feedback.
"""

    response = llm.invoke(prompt)

    return response.content