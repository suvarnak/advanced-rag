from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class PDFRAGSystem:
    def __init__(self, pdf_folder: str, index_path: str = "faiss_index"):
        self.pdf_folder = pdf_folder
        self.index_path = index_path
        # Using a more powerful embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.llm = OllamaLLM(
            model="llama3.2", 
            temperature=0.1,
            max_tokens=512,
            context_window=4096
        )
        self.vector_store = None

    def _create_text_splitter(self):
        """Create an advanced text splitter with better chunk handling"""
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )

    def _extract_metadata(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Extract metadata from PDF documents"""
        return {
            "source": str(pdf_path),
            "page": page_num,
            "filename": pdf_path.name,
            "created_at": os.path.getctime(pdf_path)
        }

    def create_index(self):
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        text_splitter = self._create_text_splitter()
        documents: List[Document] = []

        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            for page in pages:
                # Enhance metadata
                metadata = self._extract_metadata(pdf_file, page.metadata["page"])
                chunks = text_splitter.split_text(page.page_content)
                
                # Create documents with enhanced metadata
                docs = [
                    Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            "chunk_id": i,
                            "chunk_size": len(chunk)
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                documents.extend(docs)

        print(f"{len(documents)} chunks created from {len(pdf_files)} PDF files.")
        
        # Create vector store with improved similarity search
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        self.vector_store.save_local(self.index_path)
        print(f"Index created and saved to {self.index_path}")

    def load_index(self):
        if not os.path.exists(self.index_path):
            self.create_index()
            # raise FileNotFoundError(f"Index file {self.index_path} not found.")
        allow_dangerous_deserialization: bool = False,
        self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Index loaded from {self.index_path}")


    def query(self, question: str) -> Dict[str, Any]:
        if self.vector_store is None:
            raise ValueError("Index not loaded. Please load the index before querying.")

        # Create a contextual compression retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 8,
                "score_threshold": 0.5,
            }
        )

        # Add contextual compression for better context selection
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # Enhanced prompt template with better context utilization
        prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question. 
            If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Only use information from the provided context
            2. Cite the source documents (page numbers) in your answer
            3. If multiple sources conflict, point out the discrepancy
            
            Answer: """,
            input_variables=["context", "question"]
        )

        # Create QA chain with advanced retrieval
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        # Get response with metadata
        result = qa_chain({"query": question})
        
        # Format response with sources
        response = {
            "answer": result["result"],
            "sources": [
                {
                    "filename": doc.metadata["filename"],
                    "page": doc.metadata["page"],
                    "chunk_id": doc.metadata["chunk_id"]
                }
                for doc in result["source_documents"]
            ]
        }

        return response

    # ... rest of the existing methods ...

def main():
    pdf_folder = "data"
    pdf_rag_system = PDFRAGSystem(pdf_folder)
    pdf_rag_system.load_index()
    question = "who is Marcus Aurelius?"
    #"What is the main topic of the PDF documents?"
    response = pdf_rag_system.query(question)
    
    print("\nAnswer:", response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source['filename']}, Page {source['page']}")

if __name__ == "__main__":
    main()