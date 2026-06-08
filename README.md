# RAG NLP Application

A Retrieval-Augmented Generation (RAG) pipeline that retrieves context 
from unstructured documents and generates personalised outputs.

## Tools & Technologies
- Python, LangChain
- HuggingFace Embeddings (sentence-transformers)
- FAISS Vector Store
- HuggingFace LLM (flan-t5-base)

## How It Works
1. Loads unstructured documents
2. Splits documents into chunks
3. Creates vector embeddings using HuggingFace
4. Stores embeddings in FAISS vector store
5. Retrieves relevant context for each query
6. Generates personalised output using LLM

## Use Case
- Automated personalised document generation
- Knowledge retrieval from large document collections
- Question answering over custom documents

## Author
Prasanth Goud Teegala — AI Data Engineer
