"""Этап 5: владелец документа (RBAC) + append-only аудит.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-11
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("owner_sub", sa.String(64), nullable=True))
    op.create_index("ix_documents_owner_sub", "documents", ["owner_sub"])

    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_sub", sa.String(64), nullable=False),
        sa.Column("username", sa.String(128), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("object_type", sa.String(32), nullable=True),
        sa.Column("object_id", sa.String(64), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_audit_log_ts", "audit_log", ["ts"])
    # append-only: приложение не получает UPDATE/DELETE (на уровне БД — для роли rag)
    op.execute("REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC")


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_index("ix_documents_owner_sub", table_name="documents")
    op.drop_column("documents", "owner_sub")
