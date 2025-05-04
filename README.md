# PDF RAG System

[... installation section ...]

## Features and Usage

### Basic PDF RAG

The basic RAG system provides simple document querying:

```python
from pdf_rag import PDFRAGSystem

# Initialize basic RAG
rag = PDFRAGSystem(
    pdf_folder="data",
    index_path="basic_index"
)

# Create and load index
rag.create_index()
rag.load_index()

# Simple query
question = "What is the main topic of the documents?"
answer = rag.query(question)
print(answer)
```

### Advanced PDF RAG

The advanced system includes enhanced features:

```python
from pdf_rag import PDFRAGSystem

# Initialize with advanced settings
system = PDFRAGSystem(
    pdf_folder="data",
    index_path="advanced_index",
    config={
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retriever_k": 8,
        "score_threshold": 0.7,
        "temperature": 0.1
    }
)

# Create and load index
system.create_index()
system.load_index()

# Query with source tracking
response = system.query(
    "What are the key findings in the documents?",
    return_sources=True
)

# Print answer with sources
print("\nAnswer:", response["answer"])
print("\nSources:")
for source in response["sources"]:
    print(f"- Document: {source['filename']}")
    print(f"  Page: {source['page']}")
    print(f"  Relevance Score: {source['score']:.2f}")
```

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