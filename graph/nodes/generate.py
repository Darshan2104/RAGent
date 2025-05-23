from typing import Any, Dict

from graph.state import GraphState
from graph.chains.generation import generation_chain

def generate(state: GraphState, **kwargs: Dict[str, Any]) -> GraphState:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    print(generation)
    return {"question": question, "documents": documents,"generation": generation}