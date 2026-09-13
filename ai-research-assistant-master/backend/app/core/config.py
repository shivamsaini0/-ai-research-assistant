"""
Application settings loaded from environment variables.
Uses pydantic-settings for type-safe config.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field("auto", env="LLM_PROVIDER")
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", env="GEMINI_MODEL")
    mistral_api_key: str | None = Field(None, env="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-small-latest", env="MISTRAL_MODEL")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug: bool = Field(True, env="DEBUG")

    # ChromaDB
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("research_documents", env="CHROMA_COLLECTION_NAME")

    # Chunking
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")
    max_retrieved_chunks: int = Field(6, env="MAX_RETRIEVED_CHUNKS")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Uploads
    upload_dir: str = Field("./data/uploads", env="UPLOAD_DIR")
    max_file_size_mb: int = Field(50, env="MAX_FILE_SIZE_MB")
    allowed_extensions: str = Field("pdf,docx,txt", env="ALLOWED_EXTENSIONS")

    # Auth
    auth_db_path: str = Field("./data/auth.db", env="AUTH_DB_PATH")
    jwt_secret_key: str = Field("dev-secret-change-me", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_minutes: int = Field(1440, env="ACCESS_TOKEN_MINUTES")

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    class Config:
        env_file = (".env", "../.env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cache settings so we only read .env once."""
    return Settings()
