"""
Bonobos fashion RAG chain — ChromaDB semantic retrieval → GPT-4o styling synthesis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.retriever import BonobosRetriever, RankedResult, get_retriever
from app.rag.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_context,
    build_filter_context,
)

settings = get_settings()


@dataclass
class StyleResult:
    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    top_k_retrieved: int = 0


class BonobosRAGChain:
    """
    Chroma semantic retrieval (products + outfit guides) + GPT-4o fashion synthesis.
    """

    def __init__(self, retriever: BonobosRetriever | None = None) -> None:
        self._retriever = retriever or get_retriever()
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.openai_temperature,
        )

    async def style(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        color_family: str | None = None,
        occasion: str | None = None,
    ) -> StyleResult:
        t0 = time.monotonic()

        ranked: list[RankedResult] = self._retriever.retrieve(
            query=query,
            top_k=top_k or settings.retrieval_top_k,
            category=category,
            color_family=color_family,
            occasion=occasion,
        )

        result_dicts = [
            {
                "text": r.text,
                "metadata": r.metadata,
                "score": r.score,
                "source": r.source,
            }
            for r in ranked
        ]
        context = build_context(result_dicts)
        filter_ctx = build_filter_context(category, occasion)

        user_content = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
            filter_context=filter_ctx,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = await self._llm.ainvoke(messages)
        answer = response.content or ""

        latency_ms = int((time.monotonic() - t0) * 1000)

        sources = [
            {
                "name": r.metadata.get("name"),
                "category": r.metadata.get("category"),
                "sku": r.metadata.get("sku"),
                "source_type": r.source,
                "score": round(r.score, 4),
            }
            for r in ranked
        ]

        return StyleResult(
            query=query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            top_k_retrieved=len(ranked),
        )
