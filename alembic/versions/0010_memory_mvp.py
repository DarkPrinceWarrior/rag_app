"""Слой памяти MVP (docs/MEMORY_rev4_mem0_articles.md §3, §15.2): эпизоды
(memory_events, ground-truth) + извлечённая память (memory_items, пересобираема)
+ lineage (memory_item_sources) + журнал (memory_audit_log).

Адаптация под наш стек (§15.0):
- user_id — text (наш owner_sub / OIDC sub — строка, не uuid);
- tenant_id — uuid-константа (single-org);
- embedding — vector(1024) (Qwen3-Embedding-0.6B), HNSW индексируется;
- tsv — generated-колонка из content (как chunks.tsv), BM25-контур;
- RLS пока НЕ включаем (app-level scope-фильтр; Postgres RLS — Этап 3).

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-16
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

DIM = 1024  # Qwen3-Embedding-0.6B (профиль памяти, §3.1)


def upgrade() -> None:
    # --- memory_events: сырые эпизоды (ground-truth) ---
    op.create_table(
        "memory_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "event_type IN ('message_user','message_assistant','document_uploaded',"
            "'table_extracted','citation_click','tool_call','correction')",
            name="chk_event_type",
        ),
    )
    op.create_index(
        "idx_events_scope", "memory_events",
        ["tenant_id", "user_id", "project_id", "document_id", "thread_id"],
    )
    op.create_index("idx_events_created", "memory_events", ["created_at"])
    op.execute("CREATE INDEX idx_events_payload ON memory_events USING gin (payload)")
    op.execute(
        "CREATE INDEX idx_events_retention ON memory_events (retention_until) "
        "WHERE deleted_at IS NULL"
    )

    # --- memory_items: извлечённая память (пересобираема) ---
    op.create_table(
        "memory_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("structured", postgresql.JSONB(), nullable=True),
        sa.Column(
            "source_event_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source_document_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("sensitivity", sa.Text(), nullable=False, server_default="normal"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes", postgresql.UUID(as_uuid=True), sa.ForeignKey("memory_items.id"), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("fingerprint", sa.Text(), nullable=True),
        sa.Column("memory_provider", sa.Text(), nullable=False, server_default="internal"),
        sa.Column("external_memory_id", sa.Text(), nullable=True),
        sa.Column("provider_payload", postgresql.JSONB(), nullable=True),
        sa.Column("embedding", Vector(DIM), nullable=True),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("scope IN ('user','project','document','thread','org')", name="chk_scope"),
        sa.CheckConstraint(
            "kind IN ('preference','fact','glossary','rule','task','correction','summary')",
            name="chk_kind",
        ),
        sa.CheckConstraint("sensitivity IN ('normal','sensitive','secret')", name="chk_sensitivity"),
        sa.CheckConstraint("status IN ('active','superseded','deleted')", name="chk_status"),
    )
    op.create_index(
        "idx_items_active", "memory_items",
        ["tenant_id", "user_id", "scope"], postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "idx_items_project", "memory_items", ["project_id"],
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "idx_items_document", "memory_items", ["document_id"],
        postgresql_where=sa.text("status = 'active'"),
    )
    # BM25-контур: generated tsvector из content (RU+EN), как chunks.tsv
    op.execute(
        """
        ALTER TABLE memory_items ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('russian', coalesce(content, '')) ||
            to_tsvector('english', coalesce(content, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX idx_items_tsv ON memory_items USING gin (tsv)")
    op.execute("CREATE INDEX idx_items_emb ON memory_items USING hnsw (embedding vector_cosine_ops)")
    # идемпотентность (§3.3, §7): один активный факт на ключ
    op.execute(
        """
        CREATE UNIQUE INDEX uq_items_active_fingerprint
          ON memory_items (tenant_id, user_id, project_id, scope, kind, fingerprint)
          WHERE status = 'active' AND deleted_at IS NULL AND fingerprint IS NOT NULL
        """
    )

    # --- memory_item_sources: lineage item↔event (источник истины, каскад) ---
    op.create_table(
        "memory_item_sources",
        sa.Column(
            "item_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memory_items.id", ondelete="CASCADE"), primary_key=True,
        ),
        sa.Column(
            "event_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memory_events.id", ondelete="CASCADE"), primary_key=True,
        ),
    )
    op.create_index("idx_item_sources_event", "memory_item_sources", ["event_id"])

    # --- memory_audit_log: журнал + purge ---
    op.create_table(
        "memory_audit_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("before", postgresql.JSONB(), nullable=True),
        sa.Column("after", postgresql.JSONB(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "action IN ('create','update','delete','supersede','purge',"
            "'accept_candidate','reject_candidate','gate_block','injection_attempt')",
            name="chk_audit_action",
        ),
        sa.CheckConstraint(
            "actor IN ('system','user','admin','extractor')", name="chk_audit_actor"
        ),
    )
    op.create_index("idx_audit_scope", "memory_audit_log", ["tenant_id", "user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("memory_audit_log")
    op.drop_table("memory_item_sources")
    op.drop_table("memory_items")
    op.drop_table("memory_events")
