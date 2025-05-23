import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()
WANT_2_DO_INDEXING = os.getenv("WANT_2_DO_INDEXING", "no").lower() == "yes"
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)

if WANT_2_DO_INDEXING:
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding,
        persist_directory=os.path.join(os.path.dirname(__file__), "./chroma_db"),
    )

retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=embedding,
    persist_directory="./chroma_db").as_retriever()
