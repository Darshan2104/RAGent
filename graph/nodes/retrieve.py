from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieve---")
    question = state["question"]

    documents = retriever.invoke(question, k=3)
    return {"documents": documents, "question": question}