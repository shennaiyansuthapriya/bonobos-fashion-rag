"""
Fashion product CRUD endpoints — /api/v1/products
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.product import FashionCategory, FashionProduct, FitType

router = APIRouter(prefix="/products", tags=["products"])


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=500)
    category: FashionCategory
    sku: str | None = None
    description: str | None = None
    style_notes: str | None = None
    how_to_wear: str | None = None
    pairs_well_with: str | None = None
    fit_type: FitType | None = None
    color: str | None = None
    color_family: str | None = None
    pattern: str | None = None
    material: str | None = None
    occasion: list[str] | None = None
    season: list[str] | None = None
    price_usd: float | None = None
    sale_price_usd: float | None = None


class ProductResponse(BaseModel):
    id: UUID
    name: str
    category: str
    sku: str | None
    fit_type: str | None
    color: str | None
    price_usd: float | None
    is_indexed: bool

    class Config:
        from_attributes = True


@router.post("/", response_model=ProductResponse, status_code=201)
async def create_product(body: ProductCreate, db: AsyncSession = Depends(get_db)) -> ProductResponse:
    product = FashionProduct(**body.model_dump())
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return ProductResponse(
        id=product.id,
        name=product.name,
        category=product.category.value,
        sku=product.sku,
        fit_type=product.fit_type.value if product.fit_type else None,
        color=product.color,
        price_usd=product.price_usd,
        is_indexed=product.is_indexed,
    )


@router.get("/", response_model=list[ProductResponse])
async def list_products(
    category: str | None = Query(None),
    color_family: str | None = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[ProductResponse]:
    stmt = select(FashionProduct).limit(limit)
    if category:
        stmt = stmt.where(FashionProduct.category == category)
    if color_family:
        stmt = stmt.where(FashionProduct.color_family == color_family)
    result = await db.execute(stmt)
    products = result.scalars().all()
    return [
        ProductResponse(
            id=p.id,
            name=p.name,
            category=p.category.value,
            sku=p.sku,
            fit_type=p.fit_type.value if p.fit_type else None,
            color=p.color,
            price_usd=p.price_usd,
            is_indexed=p.is_indexed,
        )
        for p in products
    ]
