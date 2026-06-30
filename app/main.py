"""
Bonobos Fashion Discovery RAG — FastAPI application entry point.
ChromaDB semantic retrieval + GPT-4o styling synthesis.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1 import evaluation as evaluation_router
from app.api.v1 import products as products_router
from app.api.v1 import style as style_router
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="Bonobos Fashion Discovery RAG",
    description=(
        "ChromaDB semantic fashion product discovery with outfit styling recommendations. "
        "Powered by text-embedding-3-large and GPT-4o creative synthesis."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(style_router.router, prefix="/api/v1")
app.include_router(products_router.router, prefix="/api/v1")
app.include_router(evaluation_router.router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    from app.core.vectorstore import get_chroma_store
    store = get_chroma_store()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "products_indexed": str(store.product_count()),
    }
