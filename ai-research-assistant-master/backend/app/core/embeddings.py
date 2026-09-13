"""
Embedding Generation
======================
Converts text chunks into dense vector representations using
a local sentence-transformer model (no API calls needed).

🧠 AI CONCEPT — Embeddings:
    An embedding is a list of numbers (a vector) that represents
    the *meaning* of a piece of text. Texts with similar meanings
    get similar vectors — they're "close" in vector space.

    Example:
        "The sky is blue"    → [0.12, -0.45, 0.89, ...]
        "The ocean is blue"  → [0.11, -0.44, 0.91, ...]  ← similar!
        "I love pizza"       → [-0.8, 0.33, -0.12, ...] ← very different

    We use `all-MiniLM-L6-v2`:
      - 384-dimensional vectors
      - Runs locally (no API cost)
      - Fast: ~14k sentences/second on CPU
      - Surprisingly strong for semantic similarity

    When a user asks a question, we embed the question and find
    the chunks whose embeddings are nearest (cosine similarity).
    That's semantic search — finding meaning, not just keywords.
"""
import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.
    First call downloads the model (~90MB), subsequent calls use cache.
    """
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded successfully")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.
    Returns a list of 384-dimensional float vectors.
    
    Uses batch processing for efficiency.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # Unit vectors → cosine sim = dot product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query.
    Applies the same normalization as document embeddings.
    """
    return embed_texts([query])[0]
