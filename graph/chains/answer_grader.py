from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class AnswerGrader(BaseModel):
    "Binary score for a checking if answer addresses the question"
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(AnswerGrader)

system = """
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
"""

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])

answer_grader :RunnableSequence = answer_grader_prompt | structured_llm_grader 
