"""
RAG Chain — Full Pipeline Orchestrator
=========================================
Ties together retrieval + LLM generation + conversation memory.

🧠 AI CONCEPT — The Full RAG Loop:

    1. User asks a question
    2. We embed the question → query vector
    3. Hybrid search (BM25 + vector) → top K relevant chunks
    4. Build prompt: [system] + [history] + [chunks] + [question]
    5. Gemini LLM generates a grounded answer
    6. We extract source citations from the retrieved chunks
    7. Answer + citations returned to user
    8. Conversation stored in memory for next turn
    
🧠 AI CONCEPT — Conversation Memory:

    LLMs are stateless — each API call is independent.
    To simulate memory, we include past messages in the prompt.
    This is called "in-context memory" (vs. external memory stores).
    
    We keep the last N turns (configurable). Trade-off:
      - More history → better multi-turn coherence, but more tokens/cost
      - Less history → cheaper, but loses context quickly
    
    For production, you'd store history in Redis or a database.
    Here we use an in-memory dict keyed by session_id.
"""
import logging
from typing import Optional

from app.core.retriever import hybrid_retrieve, invalidate_bm25_cache
from app.core.llm import generate_answer
from app.core.vector_store import add_document, list_documents, delete_document
from app.core.ingestion import ingest_document
from app.models.schemas import Source, ChatMessage

logger = logging.getLogger(__name__)

# ─── In-memory session store ──────────────────────────────────────────────────
# In production: use Redis or a database

_sessions: dict[str, list[dict]] = {}   # session_id → list of messages


def get_session_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def append_to_session(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    # Keep only last 20 messages (10 turns) to stay within token budget
    _sessions[session_id] = _sessions[session_id][-20:]


def clear_session(session_id: str):
    _sessions.pop(session_id, None)


# ─── Main RAG Function ─────────────────────────────────────────────────────────

def run_rag_query(
    session_id: str,
    query: str,
    doc_ids: Optional[list[str]] = None,
) -> dict:
    """
    Execute the full RAG pipeline for a user query.
    
    Returns:
        {
            "answer": str,
            "sources": list[Source],
            "conversation_history": list[ChatMessage],
            "tokens_used": int,
        }
    """
    logger.info(f"RAG query | session={session_id} | query='{query[:80]}...'")

    # 1. Get conversation history for this session
    history = get_session_history(session_id)

    # 2. Retrieve relevant chunks (hybrid: BM25 + vector)
    retrieved_chunks = hybrid_retrieve(query, n_results=6, doc_ids=doc_ids)

    if not retrieved_chunks:
        answer = (
            "No relevant documents found. Please upload some documents first, "
            "or try a different question."
        )
        append_to_session(session_id, "user", query)
        append_to_session(session_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": [],
            "conversation_history": [
                ChatMessage(role=m["role"], content=m["content"])
                for m in get_session_history(session_id)
            ],
            "tokens_used": 0,
        }

    # 3. Generate answer with Gemini
    answer, tokens_used = generate_answer(
        query=query,
        context_chunks=retrieved_chunks,
        conversation_history=history,
    )

    # 4. Build source citations
    sources = []
    seen_chunks = set()
    for chunk in retrieved_chunks:
        cid = chunk["chunk_id"]
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)

        meta = chunk.get("metadata", {})
        preview = chunk["content"][:200].replace("\n", " ")

        sources.append(Source(
            doc_id=meta.get("doc_id", "unknown"),
            filename=meta.get("filename", "unknown"),
            chunk_index=meta.get("chunk_index", 0),
            content_preview=preview + ("..." if len(chunk["content"]) > 200 else ""),
            relevance_score=chunk.get("rrf_score", chunk.get("similarity", 0.0)),
        ))

    # 5. Update conversation history
    append_to_session(session_id, "user", query)
    append_to_session(session_id, "assistant", answer)

    # 6. Return everything
    return {
        "answer": answer,
        "sources": sources,
        "conversation_history": [
            ChatMessage(role=m["role"], content=m["content"])
            for m in get_session_history(session_id)
        ],
        "tokens_used": tokens_used,
    }
