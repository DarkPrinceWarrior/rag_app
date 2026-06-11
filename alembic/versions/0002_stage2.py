"""Этап 2: glossary, маршрутизация документов, BabelDOC/OOXML-экспорт, валидация.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-11
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("kind", sa.String(16), nullable=False, server_default="pdf_text"),
    )
    op.add_column("documents", sa.Column("s3_key_export_pdf", sa.String(1024), nullable=True))
    op.add_column("documents", sa.Column("s3_key_export_pdf_dual", sa.String(1024), nullable=True))
    op.add_column("documents", sa.Column("s3_key_export_source", sa.String(1024), nullable=True))

    op.add_column(
        "segments",
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("segments", sa.Column("validation", postgresql.JSONB(), nullable=True))

    op.create_table(
        "glossary",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("en_term", sa.String(256), nullable=False, unique=True),
        sa.Column("ru_term", sa.String(256), nullable=False),
        sa.Column("domain", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_glossary_en_term", "glossary", ["en_term"])


def downgrade() -> None:
    op.drop_table("glossary")
    op.drop_column("segments", "validation")
    op.drop_column("segments", "needs_review")
    op.drop_column("documents", "s3_key_export_source")
    op.drop_column("documents", "s3_key_export_pdf_dual")
    op.drop_column("documents", "s3_key_export_pdf")
    op.drop_column("documents", "kind")
