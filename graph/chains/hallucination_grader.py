from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class HallucinationGrader(BaseModel):
    binary_score: bool = Field(description="Answer is grouded in the facts, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(HallucinationGrader)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrived facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by a set facts.
"""

hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n {documents} \n\n LLM Generation: {generation}")
])

hallucination_grader :RunnableSequence = hallucination_grader_prompt | structured_llm_grader