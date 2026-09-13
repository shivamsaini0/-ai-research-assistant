# Technical Report: AI Research Assistant
## RAG-Powered Document Question Answering System

**Course**: AI/ML Engineering Assignment  
**System**: ResearchAI — End-to-End Retrieval-Augmented Generation Pipeline  
**Stack**: Python · FastAPI · ChromaDB · Streamlit · Gemini/Mistral · Docker

---

## 1. Introduction

### 1.1 Problem Statement

Large language models (LLMs) have demonstrated remarkable language understanding capabilities, but suffer from a critical limitation: **hallucination** — generating confident yet factually incorrect responses from parametric memory. This becomes especially problematic in research contexts where precision and source attribution are essential.

Traditional keyword-based search (BM25, TF-IDF) addresses retrieval but lacks semantic understanding, failing to match synonyms, paraphrases, or conceptually related content.

### 1.2 Proposed Solution

We build a **Retrieval-Augmented Generation (RAG)** pipeline that:
1. Grounds LLM responses exclusively in user-provided documents
2. Forces explicit source citation for every answer
3. Combines semantic and keyword search for superior retrieval
4. Maintains conversational context across multi-turn dialogues

### 1.3 Objectives

- Ingest heterogeneous document types (PDF, DOCX, TXT)
- Implement hybrid search (BM25 + vector similarity)
- Generate cited, grounded answers via Gemini or Mistral
- Provide a quantitative evaluation using RAGAS metrics
- Deploy as a Dockerized single-command application
- Add lightweight local authentication with user profiles

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a four-layer architecture:

```
[Streamlit UI]  →  REST API  →  [Backend]  →  [Data Layer]
    |              HTTP/JSON      FastAPI        ChromaDB + Disk + SQLite
    └─ JWT auth + user profile sidebar ───────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Document Ingestion Pipeline (`ingestion.py`)

Documents are processed in four stages:

**Parsing**: File-type-specific parsers extract raw text:
- PDF: PyMuPDF (`fitz`) — preserves page structure, handles multi-column layouts
- DOCX: `python-docx` — respects paragraph boundaries
- TXT: UTF-8 with error replacement

**Cleaning**: Normalizes whitespace, removes null bytes, collapses excess newlines.

**Chunking** (Sliding Window): Text is split into overlapping windows:
```
chunk_size    = 800 words  (~600 tokens)
chunk_overlap = 150 words  (~18.75% overlap)
step          = 650 words
```
Overlap ensures semantic continuity at chunk boundaries — a key technique for preventing loss of cross-boundary information.

**Metadata tagging**: Each chunk stores `doc_id`, `filename`, `chunk_index`, `page_number`.

#### 2.2.2 Embedding Model (`embeddings.py`)

We use `sentence-transformers/all-MiniLM-L6-v2`:
- **Dimensionality**: 384 floats per vector
- **Performance**: ~14,000 sentences/second on CPU
- **Model size**: ~90MB (downloaded once, cached locally)
- **Normalization**: Unit-length vectors enable cosine similarity via dot product

All embeddings are normalized to unit vectors, making cosine similarity equivalent to dot product — computationally cheaper and numerically stable.

#### 2.2.3 Vector Store (`vector_store.py`)

ChromaDB is used as the persistent vector database:
- **Index type**: HNSW (Hierarchical Navigable Small World graph)
- **Similarity metric**: Cosine distance
- **Storage**: Local disk persistence, survives restarts
- **Document registry**: JSON file tracking document metadata separately

ChromaDB's HNSW implementation enables approximate nearest-neighbor search in O(log N) time, making retrieval efficient even as the corpus grows.

#### 2.2.4 Hybrid Search Engine (`retriever.py`)

**The core innovation**: combining two fundamentally different retrieval paradigms.

**BM25 (Okapi BM25)**:
```
BM25(q, d) = Σ IDF(qᵢ) × (f(qᵢ,d) × (k₁+1)) / (f(qᵢ,d) + k₁ × (1-b+b×|d|/avgdl))
```
- k₁ = 1.5 (term saturation), b = 0.75 (length normalization)
- Excels at exact term matching, rare words, named entities

**Vector Search (Cosine Similarity)**:
```
similarity(q, d) = (q⃗ · d⃗) / (|q⃗| × |d⃗|)
```
- Understands synonyms and paraphrasing
- Finds semantically related content even without keyword overlap

**Reciprocal Rank Fusion (RRF)**:
```
RRF_score(d) = α × 1/(k + rank_vector(d)) + β × 1/(k + rank_BM25(d))
```
Where k=60 (damping), α=0.7 (vector weight), β=0.3 (BM25 weight).

RRF is preferred over score normalization because it is robust to score distribution differences between the two systems.

#### 2.2.5 LLM Integration (`llm.py`)

**Models**: Google Gemini or Mistral
- **Temperature**: 0.2 (low for factual accuracy)
- **Max tokens**: 2048 output tokens
- **Grounded generation**: System prompt enforces citation and prohibits hallucination

**System Prompt Design**:
```
You are an AI Research Assistant. Answer questions ONLY based on 
provided context. Always cite [Source: filename, chunk N]. 
If context is insufficient, say so explicitly.
```

This "grounded generation" technique is the primary anti-hallucination mechanism.

#### 2.2.6 Conversation Memory (`rag_chain.py`)

Multi-turn memory is implemented via **in-context injection**:
- Last 20 messages (10 turns) are prepended to each prompt
- Session keyed by UUID, stored in server memory
- Trade-off: stateless (no persistence on restart) vs. simplicity

#### 2.2.7 Authentication (`auth.py`)

A lightweight local auth layer is used for the project:
- Users register and log in with name, email, and password
- Passwords are hashed with PBKDF2 and stored in a local SQLite database
- JWT access tokens are issued on login
- The Streamlit sidebar displays the current user profile and attaches the token to protected API calls

#### 2.2.8 RAG Chain Orchestrator

```
User Query
    ↓
[Embed Query] → 384-dim vector
    ↓
[Hybrid Retrieve] → Top 6 chunks (BM25 + Vector → RRF)
    ↓
[Build Prompt] → [System] + [History] + [Context Chunks] + [Query]
    ↓
[Gemini LLM] → Generated answer
    ↓
[Extract Citations] → Source cards with relevance scores
    ↓
Return to frontend
```

---

## 3. Evaluation Methodology

### 3.1 RAGAS Framework

We evaluate using **RAGAS** (Retrieval Augmented Generation Assessment), a framework that uses an LLM-as-judge paradigm to score RAG pipeline quality:

#### Metric 1: Faithfulness
**Definition**: Fraction of answer claims that are entailed by the retrieved context.

```
Faithfulness = |{claims in answer that are supported by context}| / |{all claims in answer}|
```

Faithfulness directly measures hallucination. A score of 1.0 means every claim in the answer is traceable to retrieved context. Anything below 0.7 indicates the LLM is drawing on parametric memory.

#### Metric 2: Answer Relevancy
**Definition**: Semantic similarity between the generated answer and the original question.

Computed by generating N synthetic questions from the answer and measuring average cosine similarity with the original question:
```
Answer Relevancy = (1/N) Σ cos_sim(generated_qᵢ, original_q)
```

#### Metric 3: Context Precision
**Definition**: Proportion of retrieved chunks that are actually relevant to answering the question, weighted by rank position.

```
Context Precision@K = Σ(precisionₖ × relevanceₖ) / Σ relevanceₖ
```

Measures retrieval precision — are we fetching noise alongside signal?

#### Metric 4: Context Recall
**Definition**: Proportion of ground-truth answer claims that are present in the retrieved context. *Requires ground truth answers.*

```
Context Recall = |{claims in ground truth attributable to context}| / |{all claims in ground truth}|
```

### 3.2 Evaluation Interface

Two evaluation modes are provided:

1. **Quick Eval**: 5 generic questions run automatically against all uploaded documents
2. **Custom Eval**: User provides question-answer pairs, optionally with ground truth

Results are displayed as:
- Aggregate scores (0.0–1.0) with color-coded indicators
- Per-question breakdown table
- Overall weighted score

### 3.3 Fallback Evaluation

When RAGAS dependencies are unavailable, the system falls back to heuristic evaluation:
- Faithfulness: word overlap between answer and context
- Relevancy: answer length + citation presence bonus
- Context Precision: context-answer token overlap

---

## 4. Implementation Details

### 4.1 Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Backend API | FastAPI | 0.111 | Async-native, automatic OpenAPI docs |
| Vector DB | ChromaDB | 0.5.3 | Local, no external service, HNSW |
| Embeddings | sentence-transformers | 3.0.1 | Free, local, strong MiniLM-L6-v2 |
| LLM | Gemini or Mistral | — | Configurable grounded generation |
| PDF Parser | PyMuPDF | 1.24.5 | Best text extraction accuracy |
| Hybrid Search | rank-bm25 | 0.2.2 | Pure Python, no setup required |
| Evaluation | RAGAS | 0.1.14 | Purpose-built for RAG quality |
| Frontend | Streamlit | — | Fast UI with sidebar auth and tabs |
| Auth | SQLite + JWT | — | Local registration/login and `/me` endpoint |
| Deployment | Docker + Compose | — | Single-command deployment |

### 4.2 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Ingest PDF/DOCX/TXT file |
| POST | `/api/chat` | RAG-powered Q&A with citations |
| DELETE | `/api/chat/{session_id}` | Clear conversation history |
| GET | `/api/documents` | List all ingested documents |
| DELETE | `/api/documents/{doc_id}` | Remove document from vector store |
| POST | `/api/evaluate` | Run RAGAS evaluation on Q&A samples |
| POST | `/api/evaluate/quick` | Run 5-question self-evaluation |
| POST | `/api/auth/register` | Create a local user account |
| POST | `/api/auth/login` | Authenticate and receive JWT |
| GET | `/api/auth/me` | Return the signed-in user profile |
| GET | `/health` | Backend health check + chunk count |

### 4.3 Key Design Decisions

**Why ChromaDB over Pinecone/Weaviate?**  
ChromaDB runs locally without API keys or network calls. This is critical for Docker-based deployments and privacy. For production scale (>1M vectors), a managed service would be preferred.

**Why hybrid search over pure vector search?**  
Evaluation studies show hybrid search consistently outperforms pure vector search by 8–15% on precision@K for domain-specific corpora. BM25 excels at matching proper nouns, acronyms, and exact citations that embeddings may normalize away.

**Why in-context conversation memory over external memory?**  
For a research assistant, conversations are typically short (<20 turns). In-context memory avoids the complexity of a memory retrieval system while keeping the codebase maintainable. The 20-message window covers the common usage pattern while keeping token overhead modest.

**Why local JWT auth instead of a third-party provider?**  
Local JWT auth keeps the project self-contained, easy to run offline, and simple to explain in an academic setting. It also provides a realistic user identity layer without requiring external account setup.

---

## 5. Frontend Design

The frontend is implemented in **Streamlit** with a tabbed layout:
- **Upload**: drag-and-drop document ingestion
- **Chat**: conversation interface with citations and selected-document filtering
- **Documents**: document list, metadata, and deletion controls
- **Evaluation**: quick evaluation and custom sample scoring

The sidebar contains the authentication flow and user profile summary.

---

## 6. Deployment

```bash
# One-command deployment
docker-compose up --build

# Services
# - Backend: FastAPI on :8000
# - Streamlit: on :8501
# - ChromaDB: local persistent volume
# - Auth DB: local SQLite database
```

Docker configuration includes:
- Health check with 60s startup grace period
- Persistent volumes for ChromaDB and uploads
- Pre-downloaded embedding model (avoids cold-start delay)

---

## 7. Conclusion

ResearchAI demonstrates a production-quality RAG pipeline with:
- **Accurate retrieval** via hybrid BM25 + vector search with RRF fusion
- **Grounded generation** via Gemini or Mistral with explicit citation enforcement
- **Quantitative evaluation** via RAGAS faithfulness, relevancy, and precision metrics
- **Multi-turn coherence** via in-context conversation memory
- **User authentication** via local JWT login/register and profile display
- **One-command deployment** via Docker Compose

The system is designed for extensibility: swapping the LLM provider, the vector store, or the embedding model requires changing configuration values rather than rewriting the pipeline.

---

*Word count: ~1,600 words | Pages: ~4 | Format: Technical Report*
