"""
Semantic retriever for Bonobos fashion product discovery.
Uses LangChain Chroma + combined product+outfit results for outfit-aware recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.core.vectorstore import BonobosChromaStore, ChromaResult, get_chroma_store

settings = get_settings()


@dataclass
class RankedResult:
    doc_id: str
    score: float
    source: str  # "product" | "outfit"
    metadata: dict = field(default_factory=dict)
    text: str = ""


class BonobosRetriever:
    """
    ChromaDB semantic retrieval with combined product + outfit context.
    Products are primary; outfit guides provide styling context.
    """

    def __init__(self, store: BonobosChromaStore | None = None) -> None:
        self._store = store or get_chroma_store()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        color_family: str | None = None,
        occasion: str | None = None,
        include_outfits: bool = True,
    ) -> list[RankedResult]:
        top_k = top_k or settings.retrieval_top_k

        # Product search
        product_hits: list[ChromaResult] = self._store.search_products(
            query=query,
            top_k=top_k,
            category=category,
            color_family=color_family,
            occasion=occasion,
        )

        results: list[RankedResult] = [
            RankedResult(
                doc_id=h.doc_id,
                score=h.score,
                source="product",
                metadata=h.metadata,
                text=h.text,
            )
            for h in product_hits
        ]

        # Outfit / styling guides — inject top 3 for style context
        if include_outfits:
            outfit_hits: list[ChromaResult] = self._store.search_outfits(query=query, top_k=3)
            results.extend(
                RankedResult(
                    doc_id=h.doc_id,
                    score=h.score,
                    source="outfit",
                    metadata=h.metadata,
                    text=h.text,
                )
                for h in outfit_hits
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results


_retriever: BonobosRetriever | None = None


def get_retriever() -> BonobosRetriever:
    global _retriever
    if _retriever is None:
        _retriever = BonobosRetriever()
    return _retriever
