"""
Style service — orchestrates indexing and RAG queries for Bonobos fashion.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.vectorstore import get_chroma_store
from app.ingestion.product_loader import index_outfit, index_product
from app.models.product import FashionProduct, StyleLog
from app.rag.chain import BonobosRAGChain, StyleResult


class FashionStyleService:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._chain = BonobosRAGChain()

    async def style_search(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
        color_family: str | None = None,
        occasion: str | None = None,
    ) -> StyleResult:
        result = await self._chain.style(
            query=query,
            top_k=top_k,
            category=category,
            color_family=color_family,
            occasion=occasion,
        )

        log = StyleLog(
            query=query,
            answer=result.answer,
            recommended_skus=[s.get("sku") for s in result.sources if s.get("source_type") == "product"],
            latency_ms=result.latency_ms,
            occasion_filter=occasion,
            category_filter=category,
        )
        self._db.add(log)
        await self._db.commit()

        return result

    async def index_product_by_id(self, product_id: str) -> dict[str, Any]:
        stmt = select(FashionProduct).where(FashionProduct.id == product_id)
        result = await self._db.execute(stmt)
        product = result.scalar_one_or_none()
        if not product:
            raise ValueError(f"Product {product_id} not found")

        product_dict = {
            "id": str(product.id),
            "sku": product.sku,
            "name": product.name,
            "category": product.category.value if product.category else None,
            "description": product.description,
            "style_notes": product.style_notes,
            "how_to_wear": product.how_to_wear,
            "pairs_well_with": product.pairs_well_with,
            "fit_type": product.fit_type.value if product.fit_type else None,
            "color": product.color,
            "color_family": product.color_family,
            "pattern": product.pattern,
            "material": product.material,
            "occasion": product.occasion,
            "season": product.season,
            "price_usd": product.price_usd,
            "sale_price_usd": product.sale_price_usd,
        }

        chroma_ids = index_product(product_dict)

        await self._db.execute(
            update(FashionProduct)
            .where(FashionProduct.id == product_id)
            .values(is_indexed=True, chroma_ids=chroma_ids)
        )
        await self._db.commit()

        return {"product_id": str(product_id), "chroma_ids": chroma_ids, "indexed": True}

    async def get_index_stats(self) -> dict[str, Any]:
        store = get_chroma_store()
        return {
            "products_count": store.product_count(),
            "outfits_count": store.outfit_count(),
            "vector_store": "ChromaDB",
            "embedding_model": "text-embedding-3-large",
        }
