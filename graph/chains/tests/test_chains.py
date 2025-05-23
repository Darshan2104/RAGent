from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain


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