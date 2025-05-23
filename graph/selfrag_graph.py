from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END, START
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search

from graph.state import GraphState

# NEEDS TO ADD FEEDBACK ALSO ALONG WITH NOT ADDRESSABLE ANSWER WHILE SEARCHING FOR A WEB AGAIN, OTHERWISE IT WILL RUN IN A INFINITE LOOP......
def decide_to_generate(state):
    if state["web_search"]:
        return WEBSEARCH
    else:
        return GENERATE
    

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---Check Hallucination---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    from graph.chains.hallucination_grader import hallucination_grader
    score = hallucination_grader.invoke({"documents":documents,"generation":generation})

    if hallucination_grader := score.binary_score:
        print("----Decision : GENERATION IS GROUNDED IN DOCUMENTs----")
        from graph.chains.answer_grader import answer_grader
        score = answer_grader.invoke({"question":question,"generation":generation})
        if answer_grader := score.binary_score:
            print("----Decision : GENERATION ADDRESSES the QUESTION----")
            return "userful"
        else:
            print("----Decision : GENERATION DOES NOT ADDRESSES the QUESTION----")
            return "not useful"
    else:
        print("----Decision : GENERATION IS NOT GROUNDED IN DOCUMENT----")
        return "not supported"
    
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(GENERATE,generate)
workflow.add_node(WEBSEARCH,web_search)

workflow.add_edge(START, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH:WEBSEARCH, GENERATE:GENERATE}
    )

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported":GENERATE,
        "not useful":WEBSEARCH,
        "userful":END
    })
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

self_rag_app = workflow.compile()


self_rag_app.get_graph().draw_mermaid_png(output_file_path="SelfRAG.png")    

