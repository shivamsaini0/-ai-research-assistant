"""Upload endpoint — ingest documents into the vector store."""
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
import aiofiles

from app.core.config import get_settings
from app.core.auth import get_current_user
from app.core.ingestion import ingest_document
from app.core.vector_store import add_document
from app.core.retriever import invalidate_bm25_cache
from app.models.schemas import UploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _current_user: dict = Depends(get_current_user),
):
    """
    Upload a PDF, DOCX, or TXT file for ingestion into the RAG pipeline.
    The file is parsed, chunked, embedded, and stored in ChromaDB.
    """
    settings = get_settings()

    # Validate file type
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {settings.allowed_extensions_list}"
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB"
        )

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = upload_dir / safe_name

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    try:
        # Ingest: parse + chunk
        chunks, metadata = ingest_document(file_path, file.filename)

        # Embed + store in ChromaDB
        add_document(chunks, metadata)

        # Invalidate BM25 cache so it's rebuilt with the new doc
        invalidate_bm25_cache()

        return UploadResponse(
            success=True,
            doc_id=metadata.doc_id,
            filename=file.filename,
            total_chunks=metadata.total_chunks,
            message=f"Successfully ingested '{file.filename}' into {metadata.total_chunks} chunks.",
        )

    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
