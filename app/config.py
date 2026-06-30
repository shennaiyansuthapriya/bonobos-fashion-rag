from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "testing", "production"] = "development"
    app_name: str = "bonobos-fashion-rag"
    app_version: str = "1.0.0"
    debug: bool = True

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-2024-08-06"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_temperature: float = 0.1  # Slightly higher for creative styling

    chroma_persist_dir: str = "./chroma_store"
    chroma_collection_products: str = "bonobos_products"
    chroma_collection_outfits: str = "bonobos_outfits"

    database_url: str = "postgresql+asyncpg://bonobos_user:bonobos_pass@localhost:5439/bonobos_db"
    sync_database_url: str = "postgresql://bonobos_user:bonobos_pass@localhost:5439/bonobos_db"
    redis_url: str = "redis://localhost:6385/0"

    retrieval_top_k: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50

    allowed_origins: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
