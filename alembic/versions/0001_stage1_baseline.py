"""Этап 1: documents + segments (baseline).

На БД, созданной этапом 1 через create_all, применять `alembic stamp 0001`.

Revision ID: 0001
Revises:
Create Date: 2026-06-11
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_DOCUMENT_STATUS = sa.Enum(
    "uploaded",
    "parsing",
    "parsed",
    "translating",
    "translated",
    "exporting",
    "done",
    "error",
    name="document_status",
)
_SEGMENT_KIND = sa.Enum(
    "heading", "paragraph", "table", "equation", "image", name="segment_kind"
)


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", _DOCUMENT_STATUS, nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("s3_key_original", sa.String(1024), nullable=False),
        sa.Column("s3_key_content_list", sa.String(1024), nullable=True),
        sa.Column("s3_key_export_docx", sa.String(1024), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("segment_count", sa.Integer(), nullable=False),
        sa.Column("translated_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("page_idx", sa.Integer(), nullable=True),
        sa.Column("kind", _SEGMENT_KIND, nullable=False),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_segments_document_id", "segments", ["document_id"])


def downgrade() -> None:
    op.drop_table("segments")
    op.drop_table("documents")
    _SEGMENT_KIND.drop(op.get_bind(), checkfirst=True)
    _DOCUMENT_STATUS.drop(op.get_bind(), checkfirst=True)
