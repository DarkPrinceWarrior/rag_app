"""GUC-контекст для RLS (§3.7): выставляет app.tenant_id/user_id/project_id/
document_id через set_config(..., is_local=true) в транзакции запроса.

Только SET LOCAL (третий аргумент true) — не сессионный SET: при пулинге
сессионные GUC текут между клиентами (§3.7). Best-effort: при ENABLE-only RLS
(до FORCE) владелец таблиц политики обходит, поэтому ошибка set_config не должна
ронять запрос.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text as sql

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_app.rag.memory.adapter import MemoryScope

logger = logging.getLogger(__name__)

_SET_GUC = sql(
    "SELECT set_config('app.tenant_id', :t, true),"
    "       set_config('app.user_id', :u, true),"
    "       set_config('app.project_id', :p, true),"
    "       set_config('app.document_id', :d, true)"
)


async def apply_scope_guc(session: AsyncSession, scope: MemoryScope) -> None:
    """Выставить scope-контекст RLS в текущей транзакции (idempotent, best-effort)."""
    try:
        await session.execute(
            _SET_GUC,
            {
                "t": str(scope.tenant_id),
                "u": scope.user_id,
                "p": str(scope.project_id) if scope.project_id else "",
                "d": str(scope.document_id) if scope.document_id else "",
            },
        )
    except Exception as exc:  # RLS ещё не FORCE → не критично
        logger.debug("apply_scope_guc: %s", exc)
