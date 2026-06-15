"""Суммаризация старой истории чата (§ 5 п.5): инкрементальная сводка
вытесненных из окна реплик хранится на сессии.

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-14
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "chat_sessions",
        sa.Column("summary_msg_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "summary_msg_count")
    op.drop_column("chat_sessions", "summary")
