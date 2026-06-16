"""Consolidation (§7): кандидаты → memory_items, идемпотентно и с temporal.

Принятие кандидата:
- create/update: idempotent через fingerprint (уникальный индекс §3.3) — повторный
  прогон не плодит дубликаты;
- supersede/update с target: старому valid_to=now + status=superseded, новый
  ссылается через supersedes;
- delete: soft-delete целевого item.
Любое изменение → memory_audit_log; lineage item↔event → memory_item_sources.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy import text as sql

from rag_app.config import settings
from rag_app.db.models import MemoryCandidate, MemoryItem
from rag_app.rag.memory.adapter import MemoryScope
from rag_app.rag.memory.events import write_audit
from rag_app.rag.memory.rls import apply_scope_guc
from rag_app.rag.memory.service import fingerprint

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_app.rag.memory.service import MemoryService

logger = logging.getLogger(__name__)


def _scope_of(cand: MemoryCandidate) -> MemoryScope:
    return MemoryScope(
        tenant_id=cand.tenant_id,
        user_id=cand.user_id,
        project_id=cand.project_id,
        document_id=cand.document_id,
        thread_id=cand.thread_id,
    )


async def _link_sources(session: AsyncSession, item_id: uuid.UUID, event_ids: list) -> None:
    for raw in event_ids or []:
        try:
            ev_id = raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))
        except (ValueError, AttributeError):
            continue
        # ON CONFLICT — item мог быть обновлён (add_or_update), связь уже есть
        await session.execute(
            sql(
                "INSERT INTO memory_item_sources (item_id, event_id) VALUES (:i, :e)"
                " ON CONFLICT DO NOTHING"
            ),
            {"i": str(item_id), "e": str(ev_id)},
        )


async def promote_candidate(
    session: AsyncSession,
    memory: MemoryService,
    cand: MemoryCandidate,
    *,
    actor: str,
) -> MemoryItem | None:
    """Принять кандидата в боевую память. Без commit — коммитит вызывающий."""
    now = datetime.now(UTC)
    scope_obj = _scope_of(cand)
    await apply_scope_guc(session, scope_obj)
    proposed = cand.proposed or {}
    event_ids = proposed.get("source_event_ids") or []
    item: MemoryItem | None = None

    if cand.action == "delete" and cand.target_item_id is not None:
        target = await session.get(MemoryItem, cand.target_item_id)
        if target is not None and target.status == "active":
            target.status = "deleted"
            target.deleted_at = now
            await write_audit(
                session, tenant_id=cand.tenant_id, action="delete", actor=actor,
                user_id=cand.user_id, item_id=target.id, reason="candidate_delete",
            )
    else:
        content = (proposed.get("content") or "").strip()
        if not content:
            _decide(cand, "rejected", actor, now)
            return None
        kind = proposed.get("kind", "fact")
        scope = proposed.get("scope", "user")
        fp = cand.fingerprint or fingerprint(kind, scope, None, content)
        supersedes_id = None
        if cand.action in ("supersede", "update") and cand.target_item_id is not None:
            old = await session.get(MemoryItem, cand.target_item_id)
            if old is not None and old.status == "active":
                old.status = "superseded"
                old.valid_to = now
                supersedes_id = old.id

        item = await memory.adapter.add_or_update(
            session,
            scope_obj=scope_obj,
            scope=scope,
            kind=kind,
            content=content,
            sensitivity=proposed.get("sensitivity", "normal"),
            confidence=float(proposed.get("confidence", cand.confidence)),
            fingerprint=fp,
            source_event_ids=[uuid.UUID(str(e)) for e in event_ids if _is_uuid(e)],
            valid_from=now,
            supersedes=supersedes_id,
        )
        await session.flush()
        await _link_sources(session, item.id, event_ids)
        await write_audit(
            session,
            tenant_id=cand.tenant_id,
            action="supersede" if supersedes_id else "create",
            actor=actor,
            user_id=cand.user_id,
            item_id=item.id,
            after={"kind": kind, "scope": scope, "content": content[:500]},
            reason="candidate_accept",
        )

    _decide(cand, "accepted" if actor in ("user", "admin") else "auto_accepted", actor, now)
    return item


async def reject_candidate(
    session: AsyncSession, cand: MemoryCandidate, *, actor: str
) -> None:
    now = datetime.now(UTC)
    _decide(cand, "rejected", actor, now)
    await write_audit(
        session, tenant_id=cand.tenant_id, action="reject_candidate", actor=actor,
        user_id=cand.user_id, reason="candidate_reject",
    )


async def consolidate_pending(
    session: AsyncSession,
    memory: MemoryService,
    *,
    tenant_id: uuid.UUID,
    auto_threshold: float | None = None,
    limit: int = 200,
) -> int:
    """Авто-принять pending-кандидатов с confidence ≥ порога (§7 п.4). Без commit."""
    threshold = settings.memory_auto_accept_confidence if auto_threshold is None else auto_threshold
    rows = (
        await session.execute(
            select(MemoryCandidate)
            .where(
                MemoryCandidate.tenant_id == tenant_id,
                MemoryCandidate.status == "pending",
                MemoryCandidate.confidence >= threshold,
            )
            .order_by(MemoryCandidate.created_at.asc())
            .limit(limit)
        )
    ).scalars().all()
    n = 0
    for cand in rows:
        try:
            await promote_candidate(session, memory, cand, actor="extractor")
            n += 1
        except Exception as exc:
            logger.warning("consolidate: кандидат %s не принят (%s)", cand.id, exc)
    return n


async def purge_expired(session: AsyncSession, *, tenant_id: uuid.UUID) -> int:
    """Retention (§7 п.6): удалить события с истёкшим retention_until (каскад чистит
    memory_item_sources) и пометить deleted авто-items, оставшиеся без источников.
    Без commit."""
    ev = await session.execute(
        sql(
            "DELETE FROM memory_events WHERE tenant_id = CAST(:t AS uuid)"
            " AND retention_until IS NOT NULL AND retention_until < now()"
        ),
        {"t": str(tenant_id)},
    )
    await session.execute(
        sql(
            "UPDATE memory_items SET status='deleted', deleted_at=now()"
            " WHERE tenant_id = CAST(:t AS uuid) AND status='active'"
            " AND source_event_ids <> '{}'"
            " AND id NOT IN (SELECT item_id FROM memory_item_sources)"
        ),
        {"t": str(tenant_id)},
    )
    return ev.rowcount or 0


def _decide(cand: MemoryCandidate, status: str, actor: str, now: datetime) -> None:
    cand.status = status
    cand.decided_at = now
    cand.decided_by = actor


def _is_uuid(v: object) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except (ValueError, AttributeError):
        return False
