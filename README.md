# RAGent
A Python implementation of advanced RAG (Retrieval Augmented Generation) architectures.


## Types of Agentic RAG architectures
### Corrective RAG Flow
- **Query Input**: User submits a query.
- **Initial Retrieval**: Relevant context fetched from the database.
- **First Response**: Model generates an initial answer.
- **Error Check**: Response evaluated for mistakes or gaps.
- **Corrective Retrieval**: New or refined context is retrieved.
- **Regeneration**: Model produces an improved response.
- **Final Output**: Corrected answer is returned to the user.

### Implement Self-RAG
- **Query Input**: User submits a query.
- **Initial Retrieval**: Relevant documents retrieved from vector store.
- **Initial Generation**: Model generates a draft answer.
- **Self-Evaluation**: Model evaluates its own answer for accuracy and relevance.
- **Query Refinement**: Model updates or expands the query based on gaps.
- **Second Retrieval**: New or better context is fetched.
- **Final Generation**: Model produces a refined, more accurate response.

### Implement Adaptive-RAG
- **Query Input**: User submits a query.
- **query Assessment**: Based on query System decides which route to used, direct websearch of retrieval path
- Rest of the steps remains same as self-RAG



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
    ├── selfrag_graph.py        # Self-RAG implementation
    ├── adaptiverag_graph.py        # adaptive RAG implementation
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

4. For Running Tests:
```bash
pytest . -s -v
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


## References
- [Corrective RAG - Oct, 2024](https://arxiv.org/pdf/2401.15884)
- [Self-RAG - Oct, 2023](https://arxiv.org/pdf/2310.11511)
- [Adaptive RAG - Mar, 2024](https://arxiv.org/pdf/2403.14403)