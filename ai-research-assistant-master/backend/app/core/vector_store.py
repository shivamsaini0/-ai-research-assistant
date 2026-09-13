"""
Vector Store — ChromaDB Interface
===================================
Manages persistent storage and retrieval of document embeddings.

🧠 AI CONCEPT — Vector Database:
    A vector database is optimized for one thing: finding the
    K nearest vectors to a query vector. Unlike SQL which filters
    by exact values, vector DBs measure *similarity* in high-dimensional
    space using algorithms like HNSW (Hierarchical Navigable Small World).

    ChromaDB stores:
      - The embedding vector (384 floats per chunk)
      - The chunk text (called "document" in Chroma's API)
      - Metadata (doc_id, filename, page_number, etc.)

    Retrieval uses cosine similarity:
      similarity = dot(query_vec, chunk_vec) / (|query_vec| * |chunk_vec|)
    
    Since we normalize embeddings to unit length, cosine sim = dot product,
    which is fast to compute.
"""
import logging
import json
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.ingestion import TextChunk, DocumentMetadata

logger = logging.getLogger(__name__)

# ─── Singleton ChromaDB client ────────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_doc_registry: dict = {}   # In-memory registry: doc_id → DocumentMetadata dict


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        persist_path = Path(settings.chroma_persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at: {persist_path}")
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        settings = get_settings()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},   # Use cosine similarity
        )
        logger.info(f"Collection '{settings.chroma_collection_name}' ready "
                    f"({_collection.count()} existing chunks)")
    return _collection


# ─── Document Registry (JSON file backed) ─────────────────────────────────────

def _get_registry_path() -> Path:
    settings = get_settings()
    return Path(settings.chroma_persist_dir) / "doc_registry.json"


def _load_registry() -> dict:
    path = _get_registry_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_registry(registry: dict):
    path = _get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2))


# ─── Public API ───────────────────────────────────────────────────────────────

def add_document(chunks: list[TextChunk], metadata: DocumentMetadata) -> None:
    """
    Add document chunks to ChromaDB.
    Each chunk is stored with its embedding, text, and metadata.
    """
    from app.core.embeddings import embed_texts

    collection = get_collection()

    # Check if doc already exists
    registry = _load_registry()
    if metadata.doc_id in registry:
        logger.info(f"Document {metadata.doc_id} already indexed. Skipping.")
        return

    # Generate embeddings for all chunks in one batch (efficient)
    texts = [chunk.content for chunk in chunks]
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts(texts)

    # Prepare ChromaDB batch insert
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "page_number": chunk.page_number or -1,
        }
        for chunk in chunks
    ]

    # ChromaDB batch upsert (safe to call multiple times)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    # Register document
    registry[metadata.doc_id] = {
        "doc_id": metadata.doc_id,
        "filename": metadata.filename,
        "file_type": metadata.file_type,
        "file_size_bytes": metadata.file_size,
        "total_chunks": metadata.total_chunks,
        "uploaded_at": metadata.uploaded_at,
        "status": "ready",
    }
    _save_registry(registry)

    logger.info(f"Stored {len(chunks)} chunks for doc '{metadata.filename}' "
                f"(doc_id={metadata.doc_id})")


def vector_search(
    query_embedding: list[float],
    n_results: int = 6,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    🧠 AI CONCEPT — Semantic Retrieval (ANN Search):
    
    Given a query embedding, find the top-N most similar chunks.
    ChromaDB uses HNSW (Approximate Nearest Neighbor) search —
    it doesn't check every vector, it navigates a graph structure
    that clusters similar vectors together. This makes search fast
    even with millions of vectors.
    
    Optional: filter by doc_ids to search within specific documents.
    """
    collection = get_collection()

    where_filter = None
    if doc_ids and len(doc_ids) > 0:
        if len(doc_ids) == 1:
            where_filter = {"doc_id": {"$eq": doc_ids[0]}}
        else:
            where_filter = {"doc_id": {"$in": doc_ids}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Convert to similarity score: 1 - (distance/2)
            similarity = 1.0 - (distance / 2.0)
            hits.append({
                "chunk_id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": round(similarity, 4),
            })

    return hits


def list_documents() -> list[dict]:
    """Return all registered documents."""
    registry = _load_registry()
    return list(registry.values())


def delete_document(doc_id: str) -> bool:
    """Remove all chunks for a document from ChromaDB and the registry."""
    registry = _load_registry()
    if doc_id not in registry:
        return False

    collection = get_collection()

    # Delete all chunks belonging to this doc
    results = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=["documents"],
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted {len(results['ids'])} chunks for doc_id={doc_id}")

    del registry[doc_id]
    _save_registry(registry)
    return True


def get_document_count() -> int:
    """Total number of chunks stored."""
    return get_collection().count()
