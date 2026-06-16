"""RLS на таблицах памяти (docs/MEMORY_rev4_mem0_articles.md §3.7, §15.4):
второй контур изоляции поверх app-level scope-фильтра.

Фазированный rollout (безопасно для прод): здесь только ENABLE + политики на
GUC (`app.tenant_id`/`app.user_id`/`app.project_id`/`app.document_id`, ставятся
через SET LOCAL/set_config в транзакции запроса — `rag_app.rag.memory.rls`).
Под ENABLE владелец таблиц (роль приложения) RLS обходит → деплой не ломает
работу, даже если GUC где-то не выставлен. Финальный flip в FORCE + роль
воркера с BYPASSRLS — отдельным шагом после верификации (deploy/memory/PROD.md):
    ALTER TABLE memory_items FORCE ROW LEVEL SECURITY; ...

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-16
"""

from __future__ import annotations

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

_TABLES = ("memory_events", "memory_items", "memory_candidates", "memory_audit_log")

# scope-предикат через GUC (nullif — пустой GUC трактуется как «не задан»)
_SCOPE_USING = """
  tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid
  AND (
    {user_pred}
  )
  AND (
    nullif(current_setting('app.project_id', true), '') IS NULL
    OR project_id IS NULL
    OR project_id = nullif(current_setting('app.project_id', true), '')::uuid
  )
  AND (
    nullif(current_setting('app.document_id', true), '') IS NULL
    OR document_id IS NULL
    OR document_id = nullif(current_setting('app.document_id', true), '')::uuid
  )
"""

# audit_log допускает user_id IS NULL (системные записи); в остальных user строго
_USER_STRICT = "user_id = current_setting('app.user_id', true)"
_USER_NULLABLE = "user_id IS NULL OR user_id = current_setting('app.user_id', true)"


def _has_project_document(table: str) -> bool:
    # memory_audit_log имеет document_id, но не project_id — упрощаем: для audit
    # фильтруем только tenant+user
    return table in ("memory_events", "memory_items", "memory_candidates")


def upgrade() -> None:
    for t in _TABLES:
        op.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        if _has_project_document(t):
            pred = _SCOPE_USING.format(user_pred=_USER_STRICT)
        else:
            pred = (
                "tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid"
                f" AND ({_USER_NULLABLE})"
            )
        op.execute(
            f"CREATE POLICY p_{t}_scope ON {t} "
            f"USING ({pred}) WITH CHECK ({pred})"
        )


def downgrade() -> None:
    for t in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS p_{t}_scope ON {t}")
        op.execute(f"ALTER TABLE {t} DISABLE ROW LEVEL SECURITY")
