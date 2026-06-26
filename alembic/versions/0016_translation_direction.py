"""Направление перевода на документе (ТЗ §4.3): source_lang/target_lang
(en|ru|zh). По умолчанию EN→RU (server_default — для существующих строк).

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("source_lang", sa.String(8), nullable=False, server_default="en"),
    )
    op.add_column(
        "documents",
        sa.Column("target_lang", sa.String(8), nullable=False, server_default="ru"),
    )


def downgrade() -> None:
    op.drop_column("documents", "target_lang")
    op.drop_column("documents", "source_lang")
