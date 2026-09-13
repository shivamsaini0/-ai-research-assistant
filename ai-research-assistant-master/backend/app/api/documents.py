"""Documents management endpoint — list and delete ingested documents."""
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.core.vector_store import list_documents, delete_document, get_document_count
from app.core.auth import get_current_user
from app.models.schemas import DeleteResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/documents")
async def get_documents(_current_user: dict = Depends(get_current_user)):
    """List all ingested documents with their metadata."""
    docs = list_documents()
    return {
        "documents": docs,
        "total": len(docs),
        "total_chunks": get_document_count(),
    }


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str, _current_user: dict = Depends(get_current_user)):
    """Delete a document and all its chunks from the vector store."""
    from app.core.retriever import invalidate_bm25_cache
    
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    
    invalidate_bm25_cache()
    return DeleteResponse(success=True, message=f"Document '{doc_id}' deleted.")
