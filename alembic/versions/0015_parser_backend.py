"""Выбор парсера pdf_text на документе: колонка parser_backend
(null → settings.pdf_parser_backend; mineru | dots_mocr | paddle_vl).

Revision ID: 0015
Revises: 0014
Create Date: 2026-06-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("parser_backend", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "parser_backend")
