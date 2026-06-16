"""ARQ-задачи слоя памяти (§4 async-ветка, §7): извлечение кандидатов после
ответа (вне latency) и периодический consolidation.

`extract_memory` ставится из chat.py после записи ответа. `consolidate_memory` —
cron (auto-accept высокоуверенных кандидатов; purge добавляется на Этапе 3).
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.models import ChatSession, MemoryCandidate, MemoryEvent
from rag_app.rag.memory.consolidate import consolidate_pending, purge_expired
from rag_app.rag.memory.events import write_audit
from rag_app.rag.memory.extract import extract_candidates, is_injection
from rag_app.rag.memory.rls import apply_scope_guc
from rag_app.rag.memory.service import fingerprint

logger = logging.getLogger(__name__)


async def extract_memory(ctx: dict, session_id: str) -> dict:
    """Окно реплик треда → кандидаты (§6.1) → memory_candidates; инъекции
    отклоняются и логируются (§6.2); высокоуверенные сразу авто-принимаются."""
    if not settings.memory_enabled:
        return {"skipped": "memory disabled"}
    sm = ctx["sessionmaker"]
    memory = ctx["memory"]
    client = ctx["llm"]

    async with sm() as session:
        sess = await session.get(ChatSession, uuid.UUID(session_id))
        if sess is None:
            return {"skipped": "no session"}
        user_id = sess.owner_sub or "local-dev"
        scope = memory.scope_for(
            user_id, project_id=sess.folder_id, document_id=sess.document_id, thread_id=sess.id
        )
        rows = (
            (
                await session.execute(
                    select(MemoryEvent)
                    .where(
                        MemoryEvent.thread_id == sess.id,
                        MemoryEvent.event_type.in_(["message_user", "message_assistant"]),
                        MemoryEvent.deleted_at.is_(None),
                    )
                    .order_by(MemoryEvent.created_at.desc())
                    .limit(settings.memory_extract_window)
                )
            )
            .scalars()
            .all()
        )
    rows = list(reversed(rows))
    if len(rows) < 2:
        return {"skipped": "too few events"}
    event_ids = [str(r.id) for r in rows]
    transcript = "\n".join(f"{r.role}: {(r.payload or {}).get('content', '')[:600]}" for r in rows)

    candidates = await extract_candidates(client, transcript)

    inserted = injections = 0
    async with sm() as session:
        await apply_scope_guc(session, scope)
        for c in candidates:
            content = (c.get("content") or "").strip()
            if not content:
                continue
            if is_injection(content):
                injections += 1
                await write_audit(
                    session, tenant_id=scope.tenant_id, action="injection_attempt",
                    actor="extractor", user_id=user_id, reason=content[:200],
                )
                continue
            kind = c.get("kind", "fact")
            scope_kind = c.get("scope", "user")
            session.add(
                MemoryCandidate(
                    tenant_id=scope.tenant_id,
                    user_id=user_id,
                    project_id=scope.project_id,
                    document_id=scope.document_id,
                    thread_id=scope.thread_id,
                    action=c.get("action", "create"),
                    proposed={**c, "source_event_ids": event_ids},
                    confidence=float(c.get("confidence", 0.5)),
                    fingerprint=fingerprint(kind, scope_kind, None, content),
                    memory_provider="internal",
                )
            )
            inserted += 1
        await session.commit()

        if inserted:
            # авто-принять высокоуверенные сразу, не дожидаясь cron
            await consolidate_pending(session, memory, tenant_id=scope.tenant_id)
            await session.commit()

    logger.info(
        "extract_memory %s: кандидатов=%d, инъекций отклонено=%d", session_id, inserted, injections
    )
    return {"candidates": inserted, "injections": injections}


async def consolidate_memory(ctx: dict) -> dict:
    """Периодический consolidation (cron): auto-accept высокоуверенных кандидатов
    (§7). Retention/purge подключается на Этапе 3."""
    if not settings.memory_enabled:
        return {"skipped": "memory disabled"}
    sm = ctx["sessionmaker"]
    memory = ctx["memory"]
    tenant = uuid.UUID(settings.tenant_id)
    async with sm() as session:
        n = await consolidate_pending(session, memory, tenant_id=tenant)
        purged = await purge_expired(session, tenant_id=tenant)
        await session.commit()
    logger.info("consolidate_memory: принято %d, истёкших событий удалено %d", n, purged)
    return {"accepted": n, "purged": purged}
