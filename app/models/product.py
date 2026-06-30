"""
Fashion product models — Bonobos.com mirror.
Domain: Men's fashion with style, fit, occasion, and outfit recommendation context.
"""
import enum, uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import Boolean, Enum, Float, Index, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FashionCategory(str, enum.Enum):
    PANTS = "pants"
    SHIRTS = "shirts"
    SUITS = "suits"
    BLAZERS = "blazers"
    CHINOS = "chinos"
    JEANS = "jeans"
    SHORTS = "shorts"
    POLOS = "polos"
    SWEATERS = "sweaters"
    OUTERWEAR = "outerwear"
    SWIMWEAR = "swimwear"
    ACCESSORIES = "accessories"
    SHOES = "shoes"
    OTHER = "other"


class OccasionType(str, enum.Enum):
    BUSINESS_FORMAL = "business_formal"
    BUSINESS_CASUAL = "business_casual"
    SMART_CASUAL = "smart_casual"
    CASUAL = "casual"
    WEEKEND = "weekend"
    WEDDING = "wedding"
    DATE_NIGHT = "date_night"
    OUTDOOR = "outdoor"
    ATHLETIC = "athletic"


class FitType(str, enum.Enum):
    SLIM = "slim"
    ATHLETIC = "athletic"
    STRAIGHT = "straight"
    RELAXED = "relaxed"
    TAILORED = "tailored"


class FashionProduct(Base):
    __tablename__ = "fashion_products"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    sku: Mapped[Optional[str]] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    category: Mapped[FashionCategory] = mapped_column(Enum(FashionCategory), nullable=False, index=True)
    subcategory: Mapped[Optional[str]] = mapped_column(String(100))

    # Style descriptions for RAG embedding
    description: Mapped[Optional[str]] = mapped_column(Text)
    style_notes: Mapped[Optional[str]] = mapped_column(Text)
    how_to_wear: Mapped[Optional[str]] = mapped_column(Text)
    pairs_well_with: Mapped[Optional[str]] = mapped_column(Text)

    # Style metadata
    occasion: Mapped[Optional[list]] = mapped_column(JSONB, default=list)  # ["business_casual", "smart_casual"]
    fit_type: Mapped[Optional[FitType]] = mapped_column(Enum(FitType))
    color: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    color_family: Mapped[Optional[str]] = mapped_column(String(50), index=True)  # "neutral", "navy", "earth"
    pattern: Mapped[Optional[str]] = mapped_column(String(50))  # "solid", "stripe", "plaid"
    material: Mapped[Optional[str]] = mapped_column(String(200))
    season: Mapped[Optional[list]] = mapped_column(JSONB, default=list)  # ["spring", "fall"]

    # Sizing
    waist_sizes: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    inseam_sizes: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    shirt_sizes: Mapped[Optional[list]] = mapped_column(JSONB, default=list)

    price_usd: Mapped[Optional[float]] = mapped_column(Float, index=True)
    sale_price_usd: Mapped[Optional[float]] = mapped_column(Float)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)

    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    chroma_ids: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    embedding_text: Mapped[Optional[str]] = mapped_column(Text)
    rating: Mapped[Optional[float]] = mapped_column(Float)
    review_count: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_fashion_cat_occasion", "category"),
        Index("ix_fashion_color", "color_family"),
    )


class StyleLog(Base):
    __tablename__ = "style_logs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    recommended_skus: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    occasion_filter: Mapped[Optional[str]] = mapped_column(String(50))
    category_filter: Mapped[Optional[str]] = mapped_column(String(50))
