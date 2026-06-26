"""Изоляция папок по владельцу (ТЗ §4.7.1): folders.owner_sub + составной
уникальный ключ (owner_sub, name) вместо глобально-уникального name.

Существующие папки получают owner_sub=NULL (dev-папки, видны всем не-админам,
как dev-документы). Новые папки создаются с owner_sub текущего пользователя.

Revision ID: 0017
Revises: 0016
Create Date: 2026-06-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("folders", sa.Column("owner_sub", sa.String(64), nullable=True))
    op.create_index("ix_folders_owner_sub", "folders", ["owner_sub"])
    # глобально-уникальное имя → уникальное в пределах владельца
    op.drop_constraint("folders_name_key", "folders", type_="unique")
    op.create_unique_constraint("uq_folders_owner_name", "folders", ["owner_sub", "name"])


def downgrade() -> None:
    op.drop_constraint("uq_folders_owner_name", "folders", type_="unique")
    op.create_unique_constraint("folders_name_key", "folders", ["name"])
    op.drop_index("ix_folders_owner_sub", table_name="folders")
    op.drop_column("folders", "owner_sub")
