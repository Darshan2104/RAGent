from dotenv import load_dotenv
import os
from graph.graph import app
from graph.selfrag_graph import self_rag_app

load_dotenv()

if __name__ == '__main__':
    print("Starting corrective RAG...")
    ans = app.invoke(input={"question":"What is Agent memory?"})
    # ans = self_rag_app.invoke(input={"question":"Give me list of All the startup Build By Eleon Must with it's Current valuation?"})
    print("---"*20)
    print("QUESTION : ",ans["question"])
    print("WEB_SEARCH : ",ans["web_search"])
    print("ANSWER : ",ans["generation"])