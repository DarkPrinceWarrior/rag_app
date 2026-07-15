"""Проверяемая память переводов с owner/folder RLS и отзывом.

Revision ID: 0027
Revises: 0026
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "translation_memory",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("source_normalized", sa.Text(), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("approved_translation", sa.Text(), nullable=False),
        sa.Column("source_embedding", Vector(1024)),
        sa.Column("source_lang", sa.String(8), nullable=False),
        sa.Column("target_lang", sa.String(8), nullable=False),
        sa.Column("domain", sa.String(128), nullable=False, server_default="technical"),
        sa.Column("project", sa.String(256)),
        sa.Column(
            "folder_id",
            UUID(as_uuid=True),
            sa.ForeignKey("folders.id", ondelete="CASCADE"),
        ),
        sa.Column("owner_sub", sa.String(64), nullable=False),
        sa.Column("editor_sub", sa.String(64), nullable=False),
        sa.Column("editor_name", sa.String(128)),
        sa.Column("status", sa.String(16), nullable=False, server_default="candidate"),
        sa.Column(
            "segment_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("segment_versions.id", ondelete="SET NULL"),
            unique=True,
        ),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "segment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("segments.id", ondelete="SET NULL"),
        ),
        sa.Column("approved_by_sub", sa.String(64)),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_by_sub", sa.String(64)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("revocation_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('candidate', 'approved', 'revoked')",
            name="ck_translation_memory_status",
        ),
        sa.CheckConstraint(
            "source_lang <> target_lang",
            name="ck_translation_memory_language_pair",
        ),
        sa.CheckConstraint(
            "source_hash ~ '^[0-9a-f]{64}$'",
            name="ck_translation_memory_source_hash",
        ),
        sa.CheckConstraint(
            "length(source_normalized) > 0 AND length(approved_translation) > 0"
            " AND length(owner_sub) > 0 AND length(editor_sub) > 0",
            name="ck_translation_memory_required_text",
        ),
        sa.CheckConstraint(
            "status <> 'approved' OR (source_embedding IS NOT NULL"
            " AND approved_by_sub IS NOT NULL AND approved_at IS NOT NULL"
            " AND revoked_at IS NULL)",
            name="ck_translation_memory_approved_state",
        ),
        sa.CheckConstraint(
            "status <> 'revoked' OR (revoked_by_sub IS NOT NULL"
            " AND revoked_at IS NOT NULL AND length(revocation_reason) > 0)",
            name="ck_translation_memory_revoked_state",
        ),
    )
    op.create_index("ix_translation_memory_owner_sub", "translation_memory", ["owner_sub"])
    op.create_index(
        "ix_translation_memory_exact",
        "translation_memory",
        ["owner_sub", "folder_id", "source_lang", "target_lang", "source_hash", "status"],
    )
    op.execute(
        "CREATE INDEX ix_translation_memory_source_trgm ON translation_memory "
        "USING gin (source_normalized gin_trgm_ops)"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_translation_memory_active_exact ON translation_memory "
        "(owner_sub, COALESCE(folder_id, '00000000-0000-0000-0000-000000000000'::uuid), "
        "source_lang, target_lang, domain, COALESCE(project, ''), source_hash) "
        "WHERE status = 'approved' AND revoked_at IS NULL"
    )

    op.execute("ALTER TABLE translation_memory ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE translation_memory FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY translation_memory_owner ON translation_memory "
        "USING (current_setting('app.is_admin', true) = 'on' "
        "OR owner_sub = current_setting('app.user_id', true)) "
        "WITH CHECK (current_setting('app.is_admin', true) = 'on' "
        "OR owner_sub = current_setting('app.user_id', true))"
    )
    op.execute(
        """
        DO $grant$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON translation_memory TO rag_api;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON translation_memory TO rag_worker;
          END IF;
        END
        $grant$;
        """
    )


def downgrade() -> None:
    op.drop_table("translation_memory")
