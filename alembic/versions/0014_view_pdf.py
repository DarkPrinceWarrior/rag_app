"""PDF-рендер OOXML для просмотра «как в Microsoft»: колонки s3_key_view_orig /
s3_key_view_ru (оригинал и перевод docx/xlsx/pptx → PDF через LibreOffice).

Revision ID: 0014
Revises: 0013
Create Date: 2026-06-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("s3_key_view_orig", sa.String(1024), nullable=True))
    op.add_column("documents", sa.Column("s3_key_view_ru", sa.String(1024), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "s3_key_view_ru")
    op.drop_column("documents", "s3_key_view_orig")
