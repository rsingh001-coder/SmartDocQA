SmartDocQA
RAG-based Document Question Answering System
> SmartDocQA is an AI-powered document question answering application that allows users to upload PDF documents and ask natural language questions.
> The system uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers directly from the document content.

Features
> Upload and process PDF documents
> Intelligent text chunking for long documents
> Semantic search using vector embeddings
> Context-aware answer retrieval (RAG)
> Ask questions in natural language
> Clean and interactive Streamlit UI
> Secure API key management using .env

How It Works (RAG Pipeline)
> PDF Ingestion
  The uploaded PDF is parsed and text is extracted page-by-page.
> Chunking
  Document text is split into manageable chunks for efficient embedding.
> Embedding
  Each chunk is converted into vector embeddings using Cohere embeddings.
> Vector Storage
  Embeddings are stored in Pinecone, a scalable vector database.
> Retrieval
  When a question is asked, the most relevant chunks are retrieved using vector similarity search.
> Answer Generation
  The retrieved context is passed to a language model to generate an accurate answer.

Tech Stack
> Python
> Streamlit – UI
> Cohere – Embeddings & LLM
> Pinecone – Vector Database
> PyMuPDF (fitz) – PDF processing
> python-dotenv – Environment variables
