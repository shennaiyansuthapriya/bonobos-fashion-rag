"""
Unit tests for Bonobos ChromaDB fashion retriever.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.core.retriever import BonobosRetriever, RankedResult
from app.core.vectorstore import ChromaResult


@pytest.fixture
def mock_product_results() -> list[ChromaResult]:
    return [
        ChromaResult(
            doc_id="p1",
            score=0.94,
            metadata={"sku": "BNB-CHINO-ATH-NVY", "name": "Athletic Fit Navy Chino", "category": "chinos", "occasion": '["business_casual"]'},
            text="Athletic fit navy chino for business casual and smart casual wear",
        ),
        ChromaResult(
            doc_id="p2",
            score=0.88,
            metadata={"sku": "BNB-OXFRD-SLIM-LBL", "name": "Light Blue Oxford Shirt", "category": "shirts"},
            text="Slim fit light blue Oxford shirt versatile business casual",
        ),
        ChromaResult(
            doc_id="p3",
            score=0.72,
            metadata={"sku": "BNB-BLZR-NAV-SLIM", "name": "Navy Blazer Slim", "category": "blazers"},
            text="Slim navy Italian stretch blazer for business casual and smart casual",
        ),
    ]


@pytest.fixture
def mock_outfit_results() -> list[ChromaResult]:
    return [
        ChromaResult(
            doc_id="o1",
            score=0.90,
            metadata={"title": "The Modern Business Casual", "occasion": "business_casual", "source": "outfit"},
            text="Business casual outfit: navy chino + light blue OCBD + blazer + loafers",
        ),
    ]


def test_retriever_combines_products_and_outfits(mock_product_results, mock_outfit_results) -> None:
    retriever = BonobosRetriever()

    with patch.object(retriever._store, "search_products", return_value=mock_product_results):
        with patch.object(retriever._store, "search_outfits", return_value=mock_outfit_results):
            results = retriever.retrieve(
                query="business casual outfit for office",
                include_outfits=True,
            )

    assert len(results) == 4  # 3 products + 1 outfit
    product_results = [r for r in results if r.source == "product"]
    outfit_results = [r for r in results if r.source == "outfit"]
    assert len(product_results) == 3
    assert len(outfit_results) == 1


def test_retriever_sorted_by_score(mock_product_results, mock_outfit_results) -> None:
    retriever = BonobosRetriever()

    with patch.object(retriever._store, "search_products", return_value=mock_product_results):
        with patch.object(retriever._store, "search_outfits", return_value=mock_outfit_results):
            results = retriever.retrieve(query="business casual")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_no_outfits(mock_product_results) -> None:
    retriever = BonobosRetriever()

    with patch.object(retriever._store, "search_products", return_value=mock_product_results):
        results = retriever.retrieve(
            query="slim chinos for work",
            include_outfits=False,
        )

    assert all(r.source == "product" for r in results)
    assert len(results) == 3


def test_retriever_occasion_filter_passed(mock_product_results) -> None:
    retriever = BonobosRetriever()

    with patch.object(retriever._store, "search_products", return_value=mock_product_results) as mock_search:
        with patch.object(retriever._store, "search_outfits", return_value=[]):
            retriever.retrieve(
                query="wedding outfit",
                occasion="wedding",
            )
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["occasion"] == "wedding"
