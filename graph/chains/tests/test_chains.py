from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, HallucinationGrader

def test_retriecal_grader_yes() -> None: 
    question = "Agent memory"
    document = retriever.invoke(question)
    doc_txt = document[1].page_content
    
    # Call the retrieval grader
    result :GradeDocuments = retrieval_grader.invoke({"document": document, "question": question})

    # Check if the result is as expected
    assert result.binary_score == "yes"


def test_retriecal_grader_no() -> None: 
    question = "What is a capital of India"
    document = retriever.invoke(question)
    doc_txt = document[1].page_content
    
    # Call the retrieval grader
    result :GradeDocuments = retrieval_grader.invoke({"document": document, "question": question})

    # Check if the result is as expected
    assert result.binary_score == "no"    



def test_generation_chain() -> None:
    question = "agent memeory"

    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_yes() -> None:
    question = "Agent memeory"
    docs = retriever.invoke(question)
    generate = generation_chain.invoke({"context": docs, "question": question})

    ans :HallucinationGrader = hallucination_grader.invoke({
        "documents":docs,
        "generation":generate
    })
    assert ans.binary_score

def test_hallucination_grader_no() -> None:
    question = "Agent memeory"
    docs = retriever.invoke(question)
    generate = "Delhin is a capital of India and India gonna take over the world, This century is of India. India will become SuperPower"

    ans :HallucinationGrader = hallucination_grader.invoke({
        "documents":docs,
        "generation":generate
    })
    assert not ans.binary_score    





