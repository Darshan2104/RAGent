# RAGent

A Python implementation of advanced RAG (Retrieval Augmented Generation) architectures focusing on Corrective RAG patterns.

## Project Structure

```
├── .env                    # Environment variables configuration
├── .env.example           # Example environment variables template
├── ingestion.py           # Basic Retrieval creation
├── main.py               # Main application entry point
├── graph.png             # Visualization of the workflow
│
├── chroma_db/            # Vector database storage
│   └── ...              # Database files
│
└── graph/               # Core LangGraph implementation
    ├── consts.py       # Constants and configurations
    ├── graph.py        # Main graph implementation
    ├── state.py        # State Sturcture of Graph
    │
    ├── chains/         # LLM Chain implementations
    │   ├── generation.py        # Text generation chains
    │   ├── retrieval_grader.py  # Retrieval quality assessment
    │   └── tests/              # Chain-specific tests
    │
    └── nodes/          # LangGraph Node 
        ├── generate.py         # Response generation Node
        ├── grade_documents.py  # Document relevance grading Node
        ├── retrieve.py         # Vector store retrieval Node
        └── web_search.py       # Web search Node 
```

## Features

### Corrective RAG Implementation
The project implements a Corrective RAG pattern with the following workflow:
1. Takes user query and performs semantic search on Vector DB
2. Evaluates content relevance using LLM
3. Falls back to web search if relevant content is not found
4. Generates comprehensive responses based on retrieved information

## Setup

1. Create a virtual environment:
```bash
python -m venv my_venv
source my_venv/bin/activate  # On Unix/macOS
```

2. Copy `.env.example` to `.env` and configure your environment variables

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

## Architecture

The project follows a modular architecture:
- `graph/`: Contains the core Graph component implementation
- `nodes/`: Individual components of the Graph pipeline
- `chains/`: LLM chain implementations for different tasks used by Node
- `chroma_db/`: Vector database for efficient similarity search

## Todo
- Corrective RAG Flow
- Implement Self-RAG
- Implement Adaptive RAG


## References
- Corrective RAG
- Self-RAG
- Adaptive RAG