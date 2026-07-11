"""Revision-safe metadata for structured document sidecars.

Revision ID: 0023
Revises: 0022
Create Date: 2026-07-11
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

_OWNER_PREDICATE = (
    "current_setting('app.is_admin', true) = 'on'"
    " OR EXISTS (SELECT 1 FROM documents d"
    " WHERE d.id = document_structured_artifacts.document_id"
    " AND (d.owner_sub IS NULL"
    " OR d.owner_sub = current_setting('app.user_id', true)))"
)


def upgrade() -> None:
    op.create_table(
        "document_structured_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parse_revision", sa.Integer(), nullable=False),
        sa.Column("page_idx", sa.Integer(), nullable=False),
        sa.Column("artifact_type", sa.String(16), nullable=False),
        sa.Column("backend", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("artifact_key", sa.String(1024), nullable=True, unique=True),
        sa.Column("content_sha256", sa.String(64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("summary", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("parse_revision >= 0", name="ck_structured_artifact_revision"),
        sa.CheckConstraint("page_idx >= 0", name="ck_structured_artifact_page"),
        sa.CheckConstraint(
            "artifact_type IN ('kie', 'chart', 'diagram')",
            name="ck_structured_artifact_type",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'ready', 'error', 'superseded')",
            name="ck_structured_artifact_status",
        ),
        sa.CheckConstraint(
            "schema_version >= 1",
            name="ck_structured_schema_version",
        ),
        sa.CheckConstraint(
            "request_hash ~ '^[0-9a-f]{64}$'",
            name="ck_structured_request_hash",
        ),
        sa.CheckConstraint(
            "source_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_structured_source_hash",
        ),
        sa.CheckConstraint(
            "content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_structured_content_hash",
        ),
        sa.CheckConstraint(
            "size_bytes IS NULL OR size_bytes >= 0",
            name="ck_structured_size",
        ),
        sa.CheckConstraint(
            "status <> 'ready' OR (artifact_key IS NOT NULL"
            " AND content_sha256 IS NOT NULL AND size_bytes IS NOT NULL)",
            name="ck_structured_ready_payload",
        ),
        sa.UniqueConstraint(
            "document_id",
            "parse_revision",
            "page_idx",
            "artifact_type",
            "backend",
            "request_hash",
            name="uq_structured_artifact_request",
        ),
    )
    op.create_index(
        "ix_document_structured_artifacts_document_id",
        "document_structured_artifacts",
        ["document_id"],
    )
    op.create_index(
        "ix_structured_artifact_lookup",
        "document_structured_artifacts",
        ["document_id", "parse_revision", "status", "page_idx"],
    )
    op.execute("ALTER TABLE document_structured_artifacts ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY document_structured_artifacts_owner"
        " ON document_structured_artifacts"
        f" USING ({_OWNER_PREDICATE}) WITH CHECK ({_OWNER_PREDICATE})"
    )


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS document_structured_artifacts_owner"
        " ON document_structured_artifacts"
    )
    op.execute("ALTER TABLE document_structured_artifacts DISABLE ROW LEVEL SECURITY")
    op.drop_table("document_structured_artifacts")
