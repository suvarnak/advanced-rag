 # PDF RAG System
# PDF RAG System

## Installation

1. Install uv (Requires Python 3.10+):
```powershell
pipx install uv
```

2. Clone the repository:
```powershell
git clone <repository-url>
cd advanced-rag
```

3. Create and activate virtual environment using uv:
```powershell
uv venv
.venv\Scripts\activate
```

4. Install dependencies using uv sync (reads from pyproject.toml):
```powershell
uv pip sync pyproject.toml
```

5. Install and start Ollama:
- Download from [Ollama's website](https://ollama.ai)
- Pull the Llama3.2 model:
```powershell
ollama pull llama3.2
```

Note: uv sync ensures exact dependency resolution from pyproject.toml, providing faster and more reliable package installation.

## Features and Usage

### Key Features

#### Basic RAG
- Simple PDF document loading and parsing
- Basic text chunking
- Vector store using FAISS
- Question-answering with Llama2

#### Advanced RAG
- Enhanced document chunking with semantic boundaries
- BAAI/bge-large-en-v1.5 embeddings
- Contextual compression
- Source attribution and metadata tracking
- Similarity score filtering
- Custom prompt templates
- Multi-document context handling

### Example Usage

1. Place your PDFs in the `data` folder:
```
data/
  ├── document1.pdf
  ├── document2.pdf
  └── document3.pdf
```

2. Run a script for basic pdf based simple RAG. It returns answer to simple query "What is the main topic of the PDF documents?"
```
 python .\pdf_rag.py
 
```
3. Run a script for advanced  pdf based RAG. It returns answer to query "who is Marcus Aurelius?"

```
 python .\advanced_pdf_rag.py
 
```