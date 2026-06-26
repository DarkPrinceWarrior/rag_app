"""История правок + метаданные/поиск (ТЗ §4.7.2/§4.7.3): segment_versions,
documents.source_type/project_object, триграммные индексы для поиска по имени и
объекту строительства.

Revision ID: 0019
Revises: 0018
Create Date: 2026-06-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # метаданные документа
    op.add_column(
        "documents",
        sa.Column("source_type", sa.String(8), nullable=False, server_default="file"),
    )
    op.add_column("documents", sa.Column("project_object", sa.String(256)))

    # история правок перевода (append-only)
    op.create_table(
        "segment_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "segment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("old_text", sa.Text()),
        sa.Column("new_text", sa.Text()),
        sa.Column("editor_sub", sa.String(64)),
        sa.Column("editor_name", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_segment_versions_segment_id", "segment_versions", ["segment_id"])
    op.create_index("ix_segment_versions_document_id", "segment_versions", ["document_id"])

    # триграммный поиск по имени/объекту (ТЗ §4.7.3)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX ix_documents_filename_trgm ON documents USING gin (filename gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_documents_project_object_trgm "
        "ON documents USING gin (project_object gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_project_object_trgm")
    op.execute("DROP INDEX IF EXISTS ix_documents_filename_trgm")
    op.drop_index("ix_segment_versions_document_id", table_name="segment_versions")
    op.drop_index("ix_segment_versions_segment_id", table_name="segment_versions")
    op.drop_table("segment_versions")
    op.drop_column("documents", "project_object")
    op.drop_column("documents", "source_type")
