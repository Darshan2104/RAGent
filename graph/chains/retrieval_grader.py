from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    binary_score: str = Field(description="Document relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
You are a grader assessing relevance of retrieved documents to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade as relevant.
Give a binary score of 'yes' or 'no' score indicate whether the document is relevant to the question.
"""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrived document: \n\n {document} \n\n User question: {question}")
])

retrieval_grader = grader_prompt | structured_llm_grader 