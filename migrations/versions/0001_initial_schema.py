"""initial schema

Revision ID: 0001
Revises:
Create Date: 2024-08-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fashion_products",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("sku", sa.String(50), nullable=True, unique=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column(
            "category",
            sa.Enum(
                "pants", "shirts", "suits", "blazers", "chinos", "jeans", "shorts",
                "polos", "sweaters", "outerwear", "swimwear", "accessories", "shoes", "other",
                name="fashioncategory",
            ),
            nullable=False,
        ),
        sa.Column("subcategory", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("style_notes", sa.Text, nullable=True),
        sa.Column("how_to_wear", sa.Text, nullable=True),
        sa.Column("pairs_well_with", sa.Text, nullable=True),
        sa.Column("occasion", postgresql.JSONB, nullable=True),
        sa.Column(
            "fit_type",
            sa.Enum("slim", "athletic", "straight", "relaxed", "tailored", name="fittype"),
            nullable=True,
        ),
        sa.Column("color", sa.String(100), nullable=True),
        sa.Column("color_family", sa.String(50), nullable=True),
        sa.Column("pattern", sa.String(50), nullable=True),
        sa.Column("material", sa.String(200), nullable=True),
        sa.Column("season", postgresql.JSONB, nullable=True),
        sa.Column("waist_sizes", postgresql.JSONB, nullable=True),
        sa.Column("inseam_sizes", postgresql.JSONB, nullable=True),
        sa.Column("shirt_sizes", postgresql.JSONB, nullable=True),
        sa.Column("price_usd", sa.Float, nullable=True),
        sa.Column("sale_price_usd", sa.Float, nullable=True),
        sa.Column("in_stock", sa.Boolean, default=True, nullable=False),
        sa.Column("is_indexed", sa.Boolean, default=False, nullable=False),
        sa.Column("chroma_ids", postgresql.JSONB, nullable=True),
        sa.Column("embedding_text", sa.Text, nullable=True),
        sa.Column("rating", sa.Float, nullable=True),
        sa.Column("review_count", sa.Integer, nullable=True),
    )
    op.create_index("ix_fashion_products_name", "fashion_products", ["name"])
    op.create_index("ix_fashion_cat_occasion", "fashion_products", ["category"])
    op.create_index("ix_fashion_color", "fashion_products", ["color_family"])

    op.create_table(
        "style_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("recommended_skus", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("occasion_filter", sa.String(50), nullable=True),
        sa.Column("category_filter", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("style_logs")
    op.drop_table("fashion_products")
    op.execute("DROP TYPE IF EXISTS fashioncategory")
    op.execute("DROP TYPE IF EXISTS fittype")
