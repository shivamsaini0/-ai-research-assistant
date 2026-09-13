"""
AI Research Assistant — FastAPI Application
============================================
Main entry point. Mounts all API routers and configures CORS.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api import upload, chat, documents, evaluate, auth
from app.core.config import get_settings
from app.core.auth import init_auth_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load embedding model so first request is fast."""
    logger.info("🚀 AI Research Assistant starting up...")
    settings = get_settings()
    logger.info(f"Config: chunk_size={settings.chunk_size}, "
                f"overlap={settings.chunk_overlap}, "
                f"embedding={settings.embedding_model}")

    # Pre-warm embedding model (downloads on first run)
    from app.core.embeddings import get_embedding_model
    get_embedding_model()
    logger.info("✅ Embedding model ready")

    # Initialize ChromaDB
    from app.core.vector_store import get_collection
    col = get_collection()
    logger.info(f"✅ ChromaDB ready ({col.count()} chunks in store)")

    # Initialize auth store
    init_auth_db()
    logger.info("✅ Auth database ready")

    yield   # App runs here

    logger.info("👋 AI Research Assistant shutting down")


# ─── App Setup ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Research Assistant",
        description="RAG-powered document Q&A with hybrid search and citations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend (any origin in dev; lock down in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(auth.router, prefix="/api", tags=["Auth"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(documents.router, prefix="/api", tags=["Documents"])
    app.include_router(evaluate.router, prefix="/api", tags=["Evaluation"])

    # Health check
    @app.get("/health")
    async def health():
        from app.core.vector_store import get_document_count
        return {
            "status": "healthy",
            "total_chunks": get_document_count(),
        }

    # Serve frontend static files
    frontend_path = Path(__file__).parent.parent.parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(frontend_path / "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
