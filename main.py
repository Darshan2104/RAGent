from dotenv import load_dotenv
import os
from graph.graph import app

load_dotenv()

if __name__ == '__main__':
    print("Starting corrective RAG...")
    
    app.invoke(input={"question":"What is Agent memory?"})