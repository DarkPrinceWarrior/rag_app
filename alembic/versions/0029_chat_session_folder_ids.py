"""Сохраняем область из нескольких папок RAG-чата.

Revision ID: 0029
Revises: 0028
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from alembic import op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("folder_ids", ARRAY(UUID(as_uuid=True)), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "folder_ids")
