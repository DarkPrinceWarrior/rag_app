"""Второй контур изоляции по владельцу (ТЗ §4.7.1) — RLS-контекст для
documents/chunks/segments/folders/document_translations/segment_versions.

GUC `app.user_id` / `app.is_admin` выставляется в КАЖДОЙ транзакции хуком
after_begin из contextvar-принципала:
- по умолчанию — service (доверенный контур: воркер, миграции, старт) →
  app.is_admin='on' → политика пропускает всё;
- аутентифицированный API-запрос вызывает set_principal(user_sub, is_admin) →
  политика фильтрует строки по владельцу.

КОНТУР АКТИВЕН: API ходит под ролью `rag_api` (без BYPASSRLS, не владелец таблиц)
→ политики 0020 реально фильтруют. Воркер под `rag_worker` (BYPASSRLS) — service,
проходит пайплайны. Миграции под `rag` (super). Подробности — deploy/RLS.md.
Fail-open: любая ошибка хука не ставит GUC → service-режим, запрос не падает.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass

from sqlalchemy import event, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class Principal:
    user_sub: str | None = None
    is_admin: bool = False
    service: bool = True  # доверенный контур — политика пропускает (bypass)


_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "db_principal", default=Principal()
)


def set_principal(user_sub: str | None, is_admin: bool) -> None:
    """Аутентифицированный API-запрос: принципал = пользователь (не service)."""
    _principal.set(Principal(user_sub=user_sub, is_admin=bool(is_admin), service=False))


def reset_principal() -> None:
    _principal.set(Principal())


_SET_GUC = text(
    "SELECT set_config('app.user_id', :u, true), set_config('app.is_admin', :a, true)"
)


@event.listens_for(Session, "after_begin")
def _apply_rls_guc(session, transaction, connection) -> None:  # noqa: ANN001
    p = _principal.get()
    is_admin = "on" if (p.service or p.is_admin) else "off"
    uid = "" if p.service else (p.user_sub or "")
    try:
        connection.execute(_SET_GUC, {"u": uid, "a": is_admin})
    except Exception as exc:  # fail-open: второй контур не должен ронять запросы
        logger.debug("rls guc: %s", exc)
