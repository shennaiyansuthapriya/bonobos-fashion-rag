"""
Fashion style search endpoints — /api/v1/style
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.style_service import FashionStyleService

router = APIRouter(prefix="/style", tags=["style"])


class StyleRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=10, ge=1, le=30)
    category: str | None = None
    color_family: str | None = None
    occasion: str | None = None


class StyleResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    top_k_retrieved: int


@router.post("/discover", response_model=StyleResponse)
async def discover_style(request: StyleRequest, db: AsyncSession = Depends(get_db)) -> StyleResponse:
    """
    ChromaDB semantic retrieval + GPT-4o fashion styling synthesis.
    Returns product recommendations with outfit styling advice.
    """
    service = FashionStyleService(db)
    result = await service.style_search(
        query=request.query,
        top_k=request.top_k,
        category=request.category,
        color_family=request.color_family,
        occasion=request.occasion,
    )
    return StyleResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        latency_ms=result.latency_ms,
        top_k_retrieved=result.top_k_retrieved,
    )


@router.post("/products/{product_id}/index")
async def index_product(product_id: UUID, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Index a product into ChromaDB for semantic search."""
    service = FashionStyleService(db)
    try:
        return await service.index_product_by_id(str(product_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/index/stats")
async def index_stats(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    service = FashionStyleService(db)
    return await service.get_index_stats()
