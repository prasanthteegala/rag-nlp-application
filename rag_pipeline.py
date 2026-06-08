# RAG NLP Application
# Tools: LangChain, vector embeddings, OpenAI

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Load document
def load_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Split document into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Build RAG pipeline
def build_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

# Query the pipeline
def query_pipeline(qa_chain, question):
    response = qa_chain.run(question)
    return response

# Main
if __name__ == "__main__":
    # Load and process document
    documents = load_document("sample_document.txt")
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = build_rag_pipeline(vector_store)

    # Example query
    question = "What is the main topic of this document?"
    answer = query_pipeline(qa_chain, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
