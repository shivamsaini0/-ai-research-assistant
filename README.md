🚀 Built: ResearchAI — An AI Research Assistant powered by RAG


I recently built ResearchAI, an AI-powered research assistant designed to make it easier to ask questions about documents and get grounded answers with source citations.
 
The goal was to go beyond a basic chatbot and build a complete Retrieval-Augmented Generation (RAG) system with real-world components.
 
🧠 What it can do:

📄 Upload and process PDF, DOCX, and TXT documents
🔎 Hybrid search using BM25 + vector search
⚡ Reciprocal Rank Fusion (RRF) to combine search results
🤖 Gemini or Mistral LLM integration
📚 Source citations with generated answers
💬 Multi-turn conversations with session management
🔐 JWT-based authentication
📊 RAG evaluation using metrics such as faithfulness and relevance
🐳 Dockerized deployment with persistent storage
 
🏗️ Tech Stack

Python | FastAPI | Streamlit | ChromaDB | BM25 | Sentence Transformers | Gemini | Mistral | Docker | JWT
 
🔬 RAG Pipeline

Document → Parsing → Chunking → Embeddings → ChromaDB + BM25 → Hybrid Retrieval → RRF Fusion → LLM → Grounded Answer + Citations
 
One of the most interesting parts was implementing hybrid retrieval instead of relying only on semantic vector search. Combining keyword-based BM25 retrieval with vector similarity helps capture both exact terminology and semantic meaning.
 
The project also includes a modular backend architecture with separate ingestion, retrieval, LLM, authentication, API, and evaluation components.
 
💡 Key takeaway: Building an AI application isn't just about connecting an LLM to a prompt. Reliable AI systems require good retrieval, grounding, evaluation, authentication, error handling, and deployment architecture.
 
I'm continuing to explore how RAG systems can be made more accurate, reliable, and production-ready.
