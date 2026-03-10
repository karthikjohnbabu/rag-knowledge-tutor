def classify_query(question: str):

    q = question.lower()

    if "exercise" in q or "practice" in q or "problem" in q:
        return "exercise"

    if "example" in q:
        return "example"

    if "interview" in q:
        return "interview"

    if "explain" in q or "what is" in q:
        return "explanation"

    return "general"