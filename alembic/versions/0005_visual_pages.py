"""§ 12.1 шаг 4: эмбеддинги страниц для визуального поиска по сканам.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-12
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

DIM = 1024  # MRL-усечение Qwen3-VL-Embedding (полный dim 4096 не лезет в HNSW)


def upgrade() -> None:
    op.create_table(
        "page_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_idx", sa.Integer(), nullable=False),
        sa.Column("emb", Vector(DIM), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.UniqueConstraint("document_id", "page_idx", name="uq_page_embeddings_doc_page"),
    )
    op.create_index("ix_page_embeddings_document_id", "page_embeddings", ["document_id"])
    op.execute("CREATE INDEX ix_page_embeddings_emb ON page_embeddings USING hnsw (emb vector_cosine_ops)")


def downgrade() -> None:
    op.drop_table("page_embeddings")
