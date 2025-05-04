from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # Import Ollama
from langchain.document_loaders import PyPDFLoader  # Import PyPDFLoader
import os
from pathlib import Path

class PDFRAGSystem:
    def __init__(self, pdf_folder: str, index_path: str = "faiss_index"):
        self.pdf_folder = pdf_folder
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize the LLM (Ollama) with a specific model
        self.llm = OllamaLLM(model="llama3.2", temperature=0.1, max_tokens=512)
        self.vector_store = None

    def create_index(self):
        # Load and split PDF documents
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            # Extract text from each page
            for page in pages:
                texts.extend(text_splitter.split_text(page.page_content))
        
        print(len(texts), "chunks created from PDF files.")
        # Create vector store
        self.vector_store = FAISS.from_texts(texts, self.embeddings)

        # Save the index to disk
        self.vector_store.save_local(self.index_path)
        print(f"Index created and saved to {self.index_path}")

    def load_index(self):
        # Load the index from disk
        self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Index loaded from {self.index_path}")   
    

    def query(self, question: str) -> str:  
        if self.vector_store is None:
            raise ValueError("Index not loaded. Please load the index before querying.")

        # Create a retrieval chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        prompt_template = PromptTemplate(
            template="Answer the question based on the context: {context}\n\nQuestion: {question}",
            input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        # Query the index
        result = qa_chain({"query": question})
        return result["result"]

    def clear_index(self):
        # Clear the index from memory
        if self.vector_store is not None:
            del self.vector_store
            self.vector_store = None
            print("Index cleared from memory.")
        else:
            print("No index to clear.")     

    def delete_index(self):
        # Delete the index from disk
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            print(f"Index deleted from {self.index_path}")
        else:
            print(f"No index found at {self.index_path} to delete.")

        
    
def main():    
    pdf_folder = "data"  
    pdf_rag_system = PDFRAGSystem(pdf_folder)
    pdf_rag_system.create_index()   
    pdf_rag_system.load_index()
    question = "What is the main topic of the PDF document?"
    answer = pdf_rag_system.query(question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()