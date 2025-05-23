from typing import Any, Dict

from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader

def grade_documents(state:GraphState) -> Dict[str,Any]:
    question = state['question']
    docs = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in docs:
        score = retrieval_grader.invoke({"document": doc.page_content, "question": question})
        if score.binary_score.lower() == "yes":
            print("--GRADE: Document is Relevant")
            filtered_docs.append(doc)
        else:
            print("--GRADE: Document is NOT Relevant")
            web_search=True
            continue
    return {"document": filtered_docs, "question": question, "web_search":web_search}