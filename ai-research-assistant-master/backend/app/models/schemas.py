"""
Pydantic schemas for all API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── Document Models ─────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""
    doc_id: str
    filename: str
    file_type: str
    total_chunks: int
    file_size_bytes: int
    uploaded_at: str
    status: str = "ready"


class UploadResponse(BaseModel):
    """Response from the /upload endpoint."""
    success: bool
    doc_id: str
    filename: str
    total_chunks: int
    message: str


class DeleteResponse(BaseModel):
    success: bool
    message: str


# ─── Auth Models ─────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class UserLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class AuthUser(BaseModel):
    id: str
    name: str
    email: str
    created_at: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


# ─── Chat / RAG Models ────────────────────────────────────────────────────────

class Source(BaseModel):
    """A source citation returned with an answer."""
    doc_id: str
    filename: str
    chunk_index: int
    content_preview: str   # First 200 chars of the chunk
    relevance_score: float


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str              # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Filter retrieval to specific docs. None = search all docs."
    )


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    session_id: str
    answer: str
    sources: list[Source]
    conversation_history: list[ChatMessage]
    tokens_used: Optional[int] = None


# ─── Evaluation Models ────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    """Request to run evaluation on a QA pair."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


class EvalResult(BaseModel):
    """Evaluation scores."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
