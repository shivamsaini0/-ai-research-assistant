"""Chat endpoint — RAG-powered question answering."""
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.core.rag_chain import run_rag_query, clear_session
from app.core.auth import get_current_user
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _current_user: dict = Depends(get_current_user)):
    """
    Ask a question against your uploaded documents.
    
    - Runs hybrid retrieval (BM25 + vector search)
    - Generates a grounded answer with Gemini
    - Returns source citations
    - Maintains conversation history per session_id
    """
    try:
        result = run_rag_query(
            session_id=request.session_id,
            query=request.message,
            doc_ids=request.document_ids,
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=result["answer"],
            sources=result["sources"],
            conversation_history=result["conversation_history"],
            tokens_used=result.get("tokens_used"),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str, _current_user: dict = Depends(get_current_user)):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"success": True, "message": f"Session '{session_id}' cleared."}
