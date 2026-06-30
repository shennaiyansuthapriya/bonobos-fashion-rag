"""
Product catalog and outfit guide ingestion for Bonobos fashion RAG.
Loads from JSON/CSV, embeds with LangChain OpenAI embeddings, upserts to ChromaDB.
"""
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.vectorstore import get_chroma_store

settings = get_settings()


def _build_product_text(product: dict[str, Any]) -> str:
    """Rich embedding text optimised for fashion semantic search."""
    parts: list[str] = []
    if product.get("name"):
        parts.append(f"Product: {product['name']}")
    if product.get("category"):
        parts.append(f"Type: {product['category']}")
    if product.get("description"):
        parts.append(product["description"])
    if product.get("style_notes"):
        parts.append(f"Style: {product['style_notes']}")
    if product.get("how_to_wear"):
        parts.append(f"How to wear: {product['how_to_wear']}")
    if product.get("pairs_well_with"):
        parts.append(f"Pairs with: {product['pairs_well_with']}")

    style_parts = []
    if product.get("fit_type"):
        style_parts.append(f"{product['fit_type']} fit")
    if product.get("color"):
        style_parts.append(product["color"])
    if product.get("pattern"):
        style_parts.append(product["pattern"])
    if product.get("material"):
        style_parts.append(product["material"])
    if style_parts:
        parts.append("Style details: " + ", ".join(style_parts))

    if product.get("occasion"):
        occasions = product["occasion"]
        if isinstance(occasions, list):
            parts.append(f"Occasion: {', '.join(occasions)}")
    if product.get("season"):
        seasons = product["season"]
        if isinstance(seasons, list):
            parts.append(f"Season: {', '.join(seasons)}")
    if product.get("price_usd"):
        parts.append(f"Price: ${product['price_usd']:.0f}")

    return ". ".join(parts)


def _build_outfit_text(outfit: dict[str, Any]) -> str:
    return (
        f"Outfit: {outfit.get('title', '')}\n"
        f"Occasion: {outfit.get('occasion', '')}\n"
        f"{outfit.get('description', '')}\n"
        f"Items: {outfit.get('items', '')}"
    )


def index_product(product: dict[str, Any]) -> list[str]:
    """Embed and upsert a single product to ChromaDB. Returns Chroma IDs."""
    store = get_chroma_store()
    text = _build_product_text(product)
    chunks = _split_chunks(text)

    texts = chunks
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "id": str(product.get("id", "")),
            "sku": product.get("sku", ""),
            "name": product.get("name", ""),
            "category": product.get("category", ""),
            "fit_type": product.get("fit_type", ""),
            "color": product.get("color", ""),
            "color_family": product.get("color_family", ""),
            "pattern": product.get("pattern", ""),
            "material": product.get("material", ""),
            "occasion": json.dumps(product.get("occasion", [])),
            "price_usd": product.get("price_usd", 0),
            "sale_price_usd": product.get("sale_price_usd", 0),
        }
    ] * len(chunks)

    store.upsert_products(texts=texts, metadatas=metadatas, ids=ids)
    return ids


def index_outfit(outfit: dict[str, Any]) -> list[str]:
    store = get_chroma_store()
    text = _build_outfit_text(outfit)
    chunks = _split_chunks(text, max_chars=1500)

    texts = chunks
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "id": str(outfit.get("id", "")),
            "title": outfit.get("title", ""),
            "occasion": outfit.get("occasion", ""),
            "season": outfit.get("season", ""),
            "source": "outfit",
        }
    ] * len(chunks)

    store.upsert_outfits(texts=texts, metadatas=metadatas, ids=ids)
    return ids


def _split_chunks(text: str, max_chars: int = 2000, overlap_chars: int = 100) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


def load_products_from_json(filepath: str) -> int:
    path = Path(filepath)
    with open(path) as f:
        products: list[dict] = json.load(f)
    count = 0
    for product in products:
        index_product(product)
        count += 1
    return count


def load_outfits_from_json(filepath: str) -> int:
    path = Path(filepath)
    with open(path) as f:
        outfits: list[dict] = json.load(f)
    count = 0
    for outfit in outfits:
        index_outfit(outfit)
        count += 1
    return count
