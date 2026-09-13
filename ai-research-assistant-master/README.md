# 🔬 ResearchAI — AI Research Assistant

> **RAG-powered document Q&A** with hybrid search, source citations, and multi-turn conversations.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-purple)](https://trychroma.com)
[![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20Mistral-orange)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)

---

## 📐 Architecture

```
┌──────────────────── Streamlit Frontend ─────────────────────────┐
│  Upload Tab │ Chat Tab │ Documents Tab │ Evaluation Tab            │
└──────────────────────────┬───────────────────────────────────────┘
                           │ REST API (HTTP)
┌──────────────────── Backend (FastAPI) ───────────────────────────┐
│                                                                   │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────┐   │
│  │  Ingestion  │   │   RAG Chain     │   │   Chat API       │   │
│  │  Pipeline   │   │                 │   │   (sessions)     │   │
│  │ PDF/DOCX/TXT│   │ Retrieve → LLM  │   │                  │   │
│  └──────┬──────┘   └────────┬────────┘   └──────────────────┘   │
│         │                   │                                     │
│  ┌──────▼───────────────────▼───────────────────────────────┐    │
│  │            Hybrid Search Engine                          │    │
│  │   BM25 (keyword) + ChromaDB (vector) → RRF Fusion       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │            Gemini or Mistral LLM layer                   │    │
│  │            Grounded generation + citations               │    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API Key or Mistral API Key ([Gemini](https://aistudio.google.com) / [Mistral](https://console.mistral.ai))
- Docker (optional, for containerized deployment)

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd ai-research-assistant

# Copy environment file and add your API key
cp .env.example backend/.env
# Edit backend/.env and set at least one model key plus JWT_SECRET_KEY
# Example:
# GEMINI_API_KEY=your_key_here
# or
# MISTRAL_API_KEY=your_key_here
# JWT_SECRET_KEY=change-this-in-production
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Backend

```bash
# From the backend/ directory
python -m app.main

# Or with uvicorn directly:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Frontend

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Open it at `http://localhost:8501`.

### 5. Sign In

Use the Streamlit sidebar to create a local account or log in. After login, the sidebar shows your user id, name, and email, and the app uses your JWT for upload/chat/evaluation requests.

### 6. Legacy Frontend

The old `frontend/` HTML/JS interface is kept only as legacy assets. The supported UI is the Streamlit app.

---

## 🐳 Docker Deployment

```bash
# From project root
cp .env.example .env
# Edit .env and set GEMINI_API_KEY or MISTRAL_API_KEY plus JWT_SECRET_KEY

docker-compose up --build
```

Backend will be available at `http://localhost:8000` and Streamlit at `http://localhost:8501`.

The Docker setup persists ChromaDB, uploaded files, and the local auth database across restarts.

Persistent data (ChromaDB + uploads) stored in Docker volumes.

---

## 📁 Project Structure

```
ai-research-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/
│   │   │   ├── upload.py        # POST /api/upload
│   │   │   ├── chat.py          # POST /api/chat
│   │   │   └── documents.py     # GET/DELETE /api/documents
│   │   ├── core/
│   │   │   ├── config.py        # Settings from .env
│   │   │   ├── ingestion.py     # Document parsing & chunking
│   │   │   ├── embeddings.py    # sentence-transformers embeddings
│   │   │   ├── vector_store.py  # ChromaDB interface
│   │   │   ├── retriever.py     # Hybrid BM25 + vector search
│   │   │   ├── llm.py           # Gemini LLM client
│   │   │   └── rag_chain.py     # Full RAG pipeline
│   │   └── models/
│   │       └── schemas.py       # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit_app.py             # Streamlit frontend
├── frontend/                    # Legacy static UI assets
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── upload.js
│       ├── chat.js
│       └── app.js
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🧠 How It Works (RAG Pipeline)

1. **Upload** — User uploads PDF/DOCX/TXT → parsed → split into 800-word overlapping chunks
2. **Embed** — Each chunk embedded using `all-MiniLM-L6-v2` (384-dimensional vectors)
3. **Store** — Vectors + text stored in ChromaDB with metadata
4. **Query** — User asks a question
5. **Retrieve** — Hybrid search:
   - Vector search (semantic similarity via cosine distance)
   - BM25 keyword search
   - Results fused with Reciprocal Rank Fusion (RRF)
6. **Generate** — Top 6 chunks sent as context to Gemini with grounded generation prompt
7. **Respond** — Answer returned with source citations and relevance scores

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | `auto`, `gemini`, or `mistral` |
| `GEMINI_API_KEY` | optional | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model used for generateContent |
| `MISTRAL_API_KEY` | optional | Mistral API key |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral chat model |
| `AUTH_DB_PATH` | `./data/auth.db` | Local SQLite user database |
| `JWT_SECRET_KEY` | `change-this-in-production` | Secret used to sign JWTs |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_MINUTES` | `1440` | JWT expiry in minutes |
| `CHUNK_SIZE` | `800` | Words per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `MAX_RETRIEVED_CHUNKS` | `6` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size |

---

## 📊 Recommended Datasets

| Dataset | Size | Link |
|---------|------|------|
| arXiv Papers | 1.7M papers | [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) |
| CORD-19 COVID Papers | 500K papers | [Kaggle](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) |
| Wikipedia Articles | Full corpus | [Kaggle](https://www.kaggle.com/datasets/jkkphys/english-wikipedia-articles-20170820-sqlite) |

---

## 🎓 Evaluation Criteria Coverage

| Criterion | Implementation |
|-----------|----------------|
| **Code Quality** | Modular structure, type hints, docstrings, error handling |
| **RAG Pipeline** | Full pipeline: parse → chunk → embed → retrieve → generate |
| **LLM Integration** | Gemini or Mistral with grounded generation + citations |
| **Hybrid Search** | BM25 + ChromaDB vector search with RRF fusion |
| **Evaluation** | RAGAS metrics (faithfulness, relevance, precision, recall) |
| **UI/UX** | Streamlit app with upload, chat, documents, and evaluation tabs |
| **Authentication** | Local JWT login/register with user profile sidebar |
| **Deployment** | Dockerized with docker-compose, health checks |

## 🔐 Authentication Flow

The app uses a minimal local authentication layer:

1. A user registers or logs in from the Streamlit sidebar.
2. The backend hashes and stores credentials in a local SQLite database.
3. A signed JWT is returned to the frontend.
4. The Streamlit client attaches the JWT to upload, chat, document, and evaluation calls.
5. The sidebar shows the current user id, name, and email via the `/api/auth/me` endpoint.

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License
