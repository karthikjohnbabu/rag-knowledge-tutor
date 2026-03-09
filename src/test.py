from dotenv import load_dotenv
load_dotenv()

from rag_chain import ask_question

while True:
    q = input("\nAsk: ")

    result = ask_question(q)

    print("\nAnswer:\n")
    print(result["result"])

    print("\nRetrieved Chunks:\n")

    for doc in result["source_documents"]:
        print(doc.page_content[:300])
        print("\n-----------------------------\n")