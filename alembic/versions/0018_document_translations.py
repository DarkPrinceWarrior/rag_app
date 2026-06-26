"""Дополнительные переводы документа (ТЗ §4.3): document_translations —
русский документ → EN/ZH по запросу, параллельно основному переводу (→ru).

Revision ID: 0018
Revises: 0017
Create Date: 2026-06-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_translations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("target_lang", sa.String(8), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="translating"),
        sa.Column("error", sa.Text()),
        sa.Column("segment_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("translated_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("needs_review_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("data", JSONB(), nullable=False, server_default="{}"),
        sa.Column("s3_key_docx", sa.String(1024)),
        sa.Column("s3_key_source", sa.String(1024)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "target_lang", name="uq_doc_translation"),
    )
    op.create_index(
        "ix_document_translations_document_id", "document_translations", ["document_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_translations_document_id", table_name="document_translations")
    op.drop_table("document_translations")
