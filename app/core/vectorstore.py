"""
ChromaDB vector store for Bonobos fashion product + outfit collections.
Uses LangChain's Chroma integration with text-embedding-3-large.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

settings = get_settings()


@dataclass
class ChromaResult:
    doc_id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""


class BonobosChromaStore:
    """
    Two Chroma collections:
      - bonobos_products : product catalog chunks
      - bonobos_outfits  : curated outfit / styling guide documents
    """

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            dimensions=3072,
        )
        self._chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._products_store: Chroma | None = None
        self._outfits_store: Chroma | None = None

    # ------------------------------------------------------------------
    # Collection access
    # ------------------------------------------------------------------

    def _get_products_store(self) -> Chroma:
        if self._products_store is None:
            self._products_store = Chroma(
                client=self._chroma_client,
                collection_name=settings.chroma_collection_products,
                embedding_function=self._embeddings,
            )
        return self._products_store

    def _get_outfits_store(self) -> Chroma:
        if self._outfits_store is None:
            self._outfits_store = Chroma(
                client=self._chroma_client,
                collection_name=settings.chroma_collection_outfits,
                embedding_function=self._embeddings,
            )
        return self._outfits_store

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_products(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        store = self._get_products_store()
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def upsert_outfits(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        store = self._get_outfits_store()
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_products(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
        color_family: str | None = None,
        occasion: str | None = None,
    ) -> list[ChromaResult]:
        store = self._get_products_store()
        where: dict[str, Any] = {}
        if category:
            where["category"] = category
        if color_family:
            where["color_family"] = color_family
        if occasion:
            where["occasion"] = {"$contains": occasion}

        kwargs: dict[str, Any] = {"k": top_k}
        if where:
            kwargs["filter"] = where

        docs_scores = store.similarity_search_with_relevance_scores(query, **kwargs)

        return [
            ChromaResult(
                doc_id=doc.metadata.get("id", ""),
                score=float(score),
                metadata=doc.metadata,
                text=doc.page_content,
            )
            for doc, score in docs_scores
        ]

    def search_outfits(self, query: str, top_k: int = 5) -> list[ChromaResult]:
        store = self._get_outfits_store()
        docs_scores = store.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            ChromaResult(
                doc_id=doc.metadata.get("id", ""),
                score=float(score),
                metadata=doc.metadata,
                text=doc.page_content,
            )
            for doc, score in docs_scores
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def product_count(self) -> int:
        col = self._chroma_client.get_or_create_collection(settings.chroma_collection_products)
        return col.count()

    def outfit_count(self) -> int:
        col = self._chroma_client.get_or_create_collection(settings.chroma_collection_outfits)
        return col.count()


_store: BonobosChromaStore | None = None


def get_chroma_store() -> BonobosChromaStore:
    global _store
    if _store is None:
        _store = BonobosChromaStore()
    return _store
