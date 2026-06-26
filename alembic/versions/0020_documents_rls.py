"""Второй контур изоляции (ТЗ §4.7.1): RLS-политики на documents/chunks/segments/
folders/document_translations/segment_versions по владельцу (app.user_id) с
обходом для service/admin (app.is_admin='on').

ENABLE (без FORCE): текущая роль `rag` владеет таблицами и имеет BYPASSRLS —
контур ДОРМАНТ (как у таблиц памяти). Активация — приложение под ролью без
BYPASSRLS (см. deploy/RLS.md). Безопасно для прода: для `rag` ничего не меняется.

Revision ID: 0020
Revises: 0019
Create Date: 2026-06-26
"""

from __future__ import annotations

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

# таблицы с owner_sub прямо на строке
_OWN = ("documents", "folders")
# таблицы, изоляция которых выводится через documents.owner_sub по document_id
_VIA_DOC = ("chunks", "segments", "document_translations", "segment_versions")
_ALL = _OWN + _VIA_DOC

_OWN_PRED = (
    "current_setting('app.is_admin', true) = 'on'"
    " OR owner_sub IS NULL"
    " OR owner_sub = current_setting('app.user_id', true)"
)


def _via_pred(t: str) -> str:
    return (
        "current_setting('app.is_admin', true) = 'on'"
        f" OR EXISTS (SELECT 1 FROM documents d WHERE d.id = {t}.document_id"
        " AND (d.owner_sub IS NULL OR d.owner_sub = current_setting('app.user_id', true)))"
    )


def upgrade() -> None:
    for t in _ALL:
        op.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
    for t in _OWN:
        op.execute(
            f"CREATE POLICY {t}_owner ON {t} USING ({_OWN_PRED}) WITH CHECK ({_OWN_PRED})"
        )
    for t in _VIA_DOC:
        # USING без WITH CHECK → то же предикат и для INSERT (service/admin проходят)
        op.execute(f"CREATE POLICY {t}_owner ON {t} USING ({_via_pred(t)})")


def downgrade() -> None:
    for t in _ALL:
        op.execute(f"DROP POLICY IF EXISTS {t}_owner ON {t}")
        op.execute(f"ALTER TABLE {t} DISABLE ROW LEVEL SECURITY")
