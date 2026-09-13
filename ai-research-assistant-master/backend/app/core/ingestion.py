"""
Document Ingestion Pipeline
============================
Handles parsing of PDF, DOCX, and TXT files, then splits them
into overlapping text chunks ready for embedding.

🧠 AI CONCEPT — Chunking Strategy:
    LLMs have a context window limit. We can't feed a 50-page PDF
    as one blob. Instead, we split documents into chunks (~800 tokens).
    
    We use OVERLAPPING chunks (150 token overlap) so that if a sentence
    is split across chunks, both chunks still have enough context.
    
    Chunk size tradeoff:
      - Too small → missing context, answers feel incomplete
      - Too large → irrelevant info mixed in, degrades retrieval precision
    
    800 tokens (~600 words) is a well-tested default for research papers.
"""
import re
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Document parsers
import fitz          # PyMuPDF — for PDFs
import docx          # python-docx — for Word docs

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

class TextChunk:
    """A single chunk of text from a document with its metadata."""
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        filename: str,
        content: str,
        chunk_index: int,
        total_chunks: int,      # filled in after splitting
        page_number: Optional[int] = None,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.filename = filename
        self.content = content
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.page_number = page_number

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number or -1,
        }


class DocumentMetadata:
    """Metadata about an ingested document."""
    def __init__(self, doc_id: str, filename: str, file_type: str,
                 file_size: int, total_chunks: int):
        self.doc_id = doc_id
        self.filename = filename
        self.file_type = file_type
        self.file_size = file_size
        self.total_chunks = total_chunks
        self.uploaded_at = datetime.now(timezone.utc).isoformat()
        self.status = "ready"


# ─── Parsers ─────────────────────────────────────────────────────────────────

def parse_pdf(file_path: Path) -> tuple[str, list[tuple[str, int]]]:
    """
    Extract text from PDF using PyMuPDF.
    Returns: (full_text, [(page_text, page_num), ...])
    """
    pages = []
    with fitz.open(str(file_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((text, page_num))
    
    full_text = "\n\n".join(text for text, _ in pages)
    return full_text, pages


def parse_docx(file_path: Path) -> str:
    """Extract text from DOCX, preserving paragraph structure."""
    doc = docx.Document(str(file_path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def parse_txt(file_path: Path) -> str:
    """Read plain text file."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def parse_document(file_path: Path) -> tuple[str, list[tuple[str, int]]]:
    """
    Route to the correct parser based on file extension.
    Returns (full_text, [(page_text, page_num), ...])
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        text = parse_docx(file_path)
        return text, [(text, 1)]
    elif ext == ".txt":
        text = parse_txt(file_path)
        return text, [(text, 1)]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove junk characters common in PDFs.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


# ─── Chunker ─────────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    doc_id: str,
    filename: str,
    page_map: Optional[list[tuple[str, int]]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[TextChunk]:
    """
    🧠 AI CONCEPT — Sliding Window Chunking:
    
    We split text using a sliding window approach:
    1. Tokenize by words (approximate token count)
    2. Take a window of `chunk_size` words
    3. Slide forward by (chunk_size - chunk_overlap) words
    4. Each chunk gets a unique ID and its position in the document
    
    This ensures semantic continuity across chunk boundaries.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Split into words (approximate tokenization)
    words = text.split()
    
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    chunk_index = 0

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) < 50:  # Skip tiny trailing chunks
            continue

        # Determine page number for this chunk (best effort)
        page_number = None
        if page_map and len(page_map) > 0:
            char_pos = len(" ".join(words[:start]))
            cumulative = 0
            for page_text, page_num in page_map:
                cumulative += len(page_text)
                if char_pos <= cumulative:
                    page_number = page_num
                    break

        chunk_id = f"{doc_id}_chunk_{chunk_index}"

        chunks.append(TextChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename,
            content=chunk_text,
            chunk_index=chunk_index,
            total_chunks=0,     # backfilled below
            page_number=page_number,
        ))
        chunk_index += 1

    # Backfill total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    logger.info(f"Split '{filename}' into {len(chunks)} chunks "
                f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def ingest_document(file_path: Path, original_filename: str) -> tuple[list[TextChunk], DocumentMetadata]:
    """
    Full ingestion pipeline for a single document:
    1. Parse (PDF/DOCX/TXT)
    2. Clean text
    3. Split into chunks
    4. Return chunks + metadata

    Raises ValueError for unsupported types or empty documents.
    """
    logger.info(f"Starting ingestion of: {original_filename}")

    # Generate stable doc_id from filename + content hash
    raw_bytes = file_path.read_bytes()
    content_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
    doc_id = f"doc_{content_hash}"

    file_size = file_path.stat().st_size
    file_type = file_path.suffix.lower().lstrip(".")

    # Parse
    raw_text, page_map = parse_document(file_path)

    if not raw_text.strip():
        raise ValueError(f"Could not extract any text from '{original_filename}'")

    # Clean
    clean = clean_text(raw_text)

    # Chunk
    chunks = split_into_chunks(
        text=clean,
        doc_id=doc_id,
        filename=original_filename,
        page_map=page_map,
    )

    if not chunks:
        raise ValueError(f"Document '{original_filename}' produced no chunks after parsing.")

    metadata = DocumentMetadata(
        doc_id=doc_id,
        filename=original_filename,
        file_type=file_type,
        file_size=file_size,
        total_chunks=len(chunks),
    )

    logger.info(f"Ingestion complete: {original_filename} → {len(chunks)} chunks, doc_id={doc_id}")
    return chunks, metadata
