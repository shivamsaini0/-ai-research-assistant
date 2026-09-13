"""
Hybrid Retriever — BM25 + Vector Search
==========================================
Combines keyword-based retrieval (BM25) with semantic vector search
for better results than either approach alone.

🧠 AI CONCEPT — Hybrid Search:

    Two retrieval methods, each with different strengths:

    1. VECTOR SEARCH (semantic):
       - Understands *meaning* — finds chunks about "machine learning"
         even if the query says "AI algorithms"
       - Weakness: can miss exact terminology (e.g., specific acronyms,
         product names, medical terms)

    2. BM25 (keyword / TF-IDF based):
       - Classic information retrieval algorithm
       - Term Frequency × Inverse Document Frequency
       - Excels at exact keyword matching
       - Weakness: doesn't understand synonyms or paraphrasing

    HYBRID = best of both worlds.
    
    We merge results using Reciprocal Rank Fusion (RRF):
      RRF score = Σ  1 / (k + rank_i)    where k=60 (damping constant)
    
    This is the same technique used in production systems like
    Elasticsearch's hybrid search and many RAG pipelines.
"""
import logging
from typing import Optional
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.embeddings import embed_query
from app.core.vector_store import vector_search, get_collection

logger = logging.getLogger(__name__)

# BM25 index — rebuilt whenever new docs are added
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: Optional[list[dict]] = None    # list of {chunk_id, content, metadata}


def _build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load all chunks from ChromaDB and build a BM25 index.
    BM25 needs the full corpus in memory (acceptable for demo scale).
    """
    collection = get_collection()
    total = collection.count()

    if total == 0:
        return None, []

    # Fetch all chunks
    all_data = collection.get(include=["documents", "metadatas"])
    corpus = []
    for i, chunk_id in enumerate(all_data["ids"]):
        corpus.append({
            "chunk_id": chunk_id,
            "content": all_data["documents"][i],
            "metadata": all_data["metadatas"][i],
        })

    # Tokenize for BM25 (simple whitespace tokenization)
    tokenized_corpus = [item["content"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"BM25 index built with {len(corpus)} chunks")
    return bm25, corpus


def _get_bm25_index() -> tuple[Optional[BM25Okapi], list[dict]]:
    """Get or build the BM25 index (lazy initialization)."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        _bm25_index, _bm25_corpus = _build_bm25_index()
    return _bm25_index, _bm25_corpus


def invalidate_bm25_cache():
    """Call this whenever a new document is added."""
    global _bm25_index, _bm25_corpus
    _bm25_index = None
    _bm25_corpus = None


def bm25_search(query: str, n_results: int = 10, doc_ids: Optional[list[str]] = None) -> list[dict]:
    """Keyword search using BM25."""
    bm25, corpus = _get_bm25_index()

    if bm25 is None or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with corpus items
    scored = list(zip(scores, corpus))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, item in scored[:n_results * 2]:   # Fetch extra, filter below
        if score <= 0:
            continue
        if doc_ids and item["metadata"].get("doc_id") not in doc_ids:
            continue
        results.append({
            "chunk_id": item["chunk_id"],
            "content": item["content"],
            "metadata": item["metadata"],
            "bm25_score": float(score),
        })
        if len(results) >= n_results:
            break

    return results


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    
    RRF score = vector_weight × (1/(k+rank_v)) + bm25_weight × (1/(k+rank_b))
    
    k=60 is standard (reduces impact of top-1 dominance).
    We weight vector results higher (0.7) since semantic search is usually
    more useful for research questions than exact keyword matching.
    """
    scores = {}   # chunk_id → combined score
    data = {}     # chunk_id → result dict

    # Score vector results
    for rank, result in enumerate(vector_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + vector_weight * (1 / (k + rank))
        data[cid] = result

    # Score BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + bm25_weight * (1 / (k + rank))
        if cid not in data:
            data[cid] = result

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused = []
    for cid in sorted_ids:
        item = data[cid].copy()
        item["rrf_score"] = round(scores[cid], 6)
        fused.append(item)

    return fused


def hybrid_retrieve(
    query: str,
    n_results: int = 6,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Main retrieval function — runs vector search + BM25, then fuses.
    
    Returns top N chunks ranked by combined RRF score,
    each with content, metadata, and source info for citations.
    """
    settings = get_settings()
    n = n_results or settings.max_retrieved_chunks

    # 1. Vector search
    query_emb = embed_query(query)
    vector_hits = vector_search(query_emb, n_results=n * 2, doc_ids=doc_ids)

    # 2. BM25 keyword search
    bm25_hits = bm25_search(query, n_results=n * 2, doc_ids=doc_ids)

    # 3. Fuse results
    if not vector_hits and not bm25_hits:
        return []

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)

    # Return top N
    top = fused[:n]
    logger.info(f"Hybrid retrieval: {len(vector_hits)} vector + {len(bm25_hits)} BM25 "
                f"→ {len(top)} fused results")
    return top
