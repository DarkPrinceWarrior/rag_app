"""Этап 3 (RAG): pgvector, chunks, чат, папки.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-11
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

DIM = 1024  # BGE-M3 dense


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "folders",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.add_column("documents", sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("documents", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("index_error", sa.Text(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "folder_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("folders.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False, server_default="section"),
        sa.Column("heading_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text_en", sa.Text(), nullable=False, server_default=""),
        sa.Column("text_ru", sa.Text(), nullable=False, server_default=""),
        sa.Column("emb_en", Vector(DIM), nullable=True),
        sa.Column("emb_ru", Vector(DIM), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    # BM25-контур (roadmap § 5 п.2): generated tsvector по RU+EN тексту
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('russian', coalesce(text_ru, '')) ||
            to_tsvector('english', coalesce(text_en, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    # HNSW — до ~5M векторов pgvector с запасом (roadmap § 1)
    op.execute("CREATE INDEX ix_chunks_emb_en ON chunks USING hnsw (emb_en vector_cosine_ops)")
    op.execute("CREATE INDEX ix_chunks_emb_ru ON chunks USING hnsw (emb_ru vector_cosine_ops)")

    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(256), nullable=False, server_default="Новый чат"),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("chunks")
    op.drop_column("documents", "folder_id")
    op.drop_column("documents", "index_error")
    op.drop_column("documents", "indexed_at")
    op.drop_column("documents", "chunk_count")
    op.drop_table("folders")
