"""Форс-OCR на документе: восстановление PDF с битым ToUnicode-cmap
текстового слоя — переразбор через MinerU -m ocr с указанным языком.

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-15
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("parse_force_ocr", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("documents", sa.Column("ocr_lang", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "ocr_lang")
    op.drop_column("documents", "parse_force_ocr")
