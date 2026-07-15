"""Обновить pgvector до 0.8.5 после замены серверного образа.

Revision ID: 0026
Revises: 0025
Create Date: 2026-07-15
"""

from __future__ import annotations

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER EXTENSION vector UPDATE TO '0.8.5'")


def downgrade() -> None:
    raise RuntimeError(
        "pgvector 0.8.5 downgrade is unsafe; restore the pre-upgrade PGDATA snapshot"
    )
