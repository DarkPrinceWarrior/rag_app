"""Визуальные эмбеддинги: полный dim 4096 — MRL-усечение ломает ранжирование
(серия Qwen3-VL-Embedding не Matryoshka-обученная). HNSW при >2000 невозможен —
последовательный скан: страниц на порядки меньше, чем чанков.

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-13
"""

from __future__ import annotations

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_page_embeddings_emb")
    op.execute("TRUNCATE page_embeddings")  # вектора старой размерности невалидны
    op.execute("ALTER TABLE page_embeddings ALTER COLUMN emb TYPE vector(4096)")


def downgrade() -> None:
    op.execute("TRUNCATE page_embeddings")
    op.execute("ALTER TABLE page_embeddings ALTER COLUMN emb TYPE vector(1024)")
    op.execute(
        "CREATE INDEX ix_page_embeddings_emb ON page_embeddings USING hnsw (emb vector_cosine_ops)"
    )
