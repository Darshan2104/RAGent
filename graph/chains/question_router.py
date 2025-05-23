from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# ... dot means this field is requried
class RouteQuery(BaseModel):
    """Route a User query to the most relevant dataSource"""
    datasource: Literal["vectorstore", "websearch"] = Field(..., description="Gien user question choose to route it to web search or a vectorstore")

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are expert at routing a user question to a vectorstore or web search.
The Vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',system),
        ('human',"{question}")
    ]
)

question_router = router_prompt | structured_llm_router



