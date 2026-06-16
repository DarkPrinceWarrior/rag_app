"""Запись эпизодов (memory_events) и журнала (memory_audit_log).

Хелперы добавляют строки в сессию БЕЗ commit — вызывающий код коммитит сам
(чтобы событие памяти жило в одной транзакции с прикладной записью).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from rag_app.db.models import MemoryAuditLog, MemoryEvent

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_app.rag.memory.adapter import MemoryScope


async def record_event(
    session: AsyncSession,
    scope: MemoryScope,
    event_type: str,
    *,
    role: str | None = None,
    payload: dict[str, Any] | None = None,
    source_message_id: uuid.UUID | None = None,
    retention_until: Any = None,
) -> MemoryEvent:
    """Сырой эпизод треда/документа (ground-truth, §3.2). Без commit."""
    ev = MemoryEvent(
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_id=scope.document_id,
        thread_id=scope.thread_id,
        event_type=event_type,
        role=role,
        payload=payload or {},
        source_message_id=source_message_id,
        retention_until=retention_until,
    )
    session.add(ev)
    return ev


async def write_audit(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    action: str,
    actor: str,
    user_id: str | None = None,
    item_id: uuid.UUID | None = None,
    event_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    reason: str | None = None,
) -> None:
    """Append-only журнал памяти (§3.6). Без commit."""
    session.add(
        MemoryAuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            item_id=item_id,
            event_id=event_id,
            document_id=document_id,
            action=action,
            actor=actor,
            before=before,
            after=after,
            reason=reason,
        )
    )
