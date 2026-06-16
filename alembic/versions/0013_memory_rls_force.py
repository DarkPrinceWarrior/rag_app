"""RLS FORCE на таблицах памяти (§3.7): default-deny второй контур теперь
применяется и к роли-владельцу приложения, не только к чужим ролям.

ПРЕДУСЛОВИЯ (см. deploy/memory/PROD.md):
- API ходит под ролью-владельцем и выставляет GUC (app.user_id и т.д.) в КАЖДОЙ
  транзакции, читающей память (apply_scope_guc вшит во все пути) — иначе под
  FORCE запрос вернёт пусто / INSERT упадёт по WITH CHECK;
- воркер (consolidation/purge ходят кросс-юзерно) должен иметь роль с BYPASSRLS.
Leakage-suite (scripts/_leakage_suite.py) должна давать 0 утечек ДО применения.

Revision ID: 0013
Revises: 0012
Create Date: 2026-06-16
"""

from __future__ import annotations

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_TABLES = ("memory_events", "memory_items", "memory_candidates", "memory_audit_log")


def upgrade() -> None:
    for t in _TABLES:
        op.execute(f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    for t in _TABLES:
        op.execute(f"ALTER TABLE {t} NO FORCE ROW LEVEL SECURITY")
