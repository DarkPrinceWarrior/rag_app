"""Привязка чатов к пользователю и папке (память, Этап 0): chat_sessions.owner_sub
+ folder_id — субстрат для thread/user/project scope слоя памяти.

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-16
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # sub владельца из OIDC-токена; NULL — сессии dev-периода (как documents.owner_sub)
    op.add_column("chat_sessions", sa.Column("owner_sub", sa.String(length=64), nullable=True))
    op.create_index("ix_chat_sessions_owner_sub", "chat_sessions", ["owner_sub"])
    # папка библиотеки → project scope треда (память §15.0); SET NULL при удалении папки
    op.add_column(
        "chat_sessions",
        sa.Column(
            "folder_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("folders.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "folder_id")
    op.drop_index("ix_chat_sessions_owner_sub", table_name="chat_sessions")
    op.drop_column("chat_sessions", "owner_sub")
