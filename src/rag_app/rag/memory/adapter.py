"""Провайдер памяти: ретрив + персистенция items.

`InternalAdapter` — нативный провайдер на нашем стеке: dense (Qwen3-Embedding,
`memory_items.embedding`) + sparse (`tsv`) → RRF → Qwen3-Reranker. Scope-фильтр
по tenant/user/project/document/thread зашит в SQL ДО поиска (§4) — второй
контур поверх gate. `MemoryAdapter`-протокол позволяет подменить движок на Mem0
(Этап 4), не трогая остальной код.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import text as sql

from rag_app.config import settings
from rag_app.db.models import MemoryItem

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_app.llm.embeddings import Embedder, Reranker

logger = logging.getLogger(__name__)

_RRF_K = 60


@dataclass
class MemoryScope:
    """Контекст доступа: чей запрос и в каком проекте/документе/треде."""

    tenant_id: uuid.UUID
    user_id: str
    project_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    thread_id: uuid.UUID | None = None


@dataclass
class MemoryHit:
    id: uuid.UUID
    scope: str
    kind: str
    content: str
    structured: dict[str, Any] | None
    confidence: float
    importance: float
    sensitivity: str
    valid_to: datetime | None
    status: str
    rerank: float = 0.0
    fused: float = 0.0


class MemoryAdapter(Protocol):
    async def search(
        self, session: AsyncSession, query: str, scope: MemoryScope, limit: int
    ) -> list[MemoryHit]: ...

    async def add_or_update(self, session: AsyncSession, **fields: Any) -> MemoryItem: ...

    async def delete(self, session: AsyncSession, item_id: uuid.UUID) -> None: ...

    async def healthcheck(self) -> dict[str, Any]: ...


# Scope-фильтр (§4, §5 п.1): tenant строго; project/document/thread совпадают
# либо item шире (user/org). NULL-контекст по ветке → её items не видны.
_SCOPE_WHERE = """
  tenant_id = CAST(:tenant AS uuid) AND user_id = :user
  AND status = 'active' AND deleted_at IS NULL
  AND (valid_to IS NULL OR valid_to > now())
  AND (
    scope IN ('user','org')
    OR (scope = 'project'  AND project_id  = CAST(:project AS uuid))
    OR (scope = 'document' AND document_id = CAST(:document AS uuid))
    OR (scope = 'thread'   AND thread_id   = CAST(:thread AS uuid))
  )
"""

_FIELDS = "id, scope, kind, content, structured, confidence, importance, sensitivity, valid_to, status"

_DENSE_SQL = f"""
SELECT {_FIELDS}, embedding <=> CAST(:qe AS vector) AS dist
FROM memory_items
WHERE {_SCOPE_WHERE} AND embedding IS NOT NULL
ORDER BY dist
LIMIT :k
"""

_SPARSE_SQL = f"""
SELECT {_FIELDS}, ts_rank(tsv, q) AS rank
FROM memory_items,
     LATERAL (SELECT websearch_to_tsquery('russian', :q)
                  || websearch_to_tsquery('english', :q) AS q) tsq
WHERE {_SCOPE_WHERE} AND tsv @@ q
ORDER BY rank DESC
LIMIT :k
"""


def _scope_params(scope: MemoryScope) -> dict[str, Any]:
    return {
        "tenant": str(scope.tenant_id),
        "user": scope.user_id,
        "project": str(scope.project_id) if scope.project_id else None,
        "document": str(scope.document_id) if scope.document_id else None,
        "thread": str(scope.thread_id) if scope.thread_id else None,
    }


def _row_to_hit(row: Any) -> MemoryHit:
    return MemoryHit(
        id=row.id,
        scope=row.scope,
        kind=row.kind,
        content=row.content,
        structured=row.structured,
        confidence=float(row.confidence),
        importance=float(row.importance),
        sensitivity=row.sensitivity,
        valid_to=row.valid_to,
        status=row.status,
    )


class InternalAdapter:
    def __init__(self, embedder: Embedder, reranker: Reranker) -> None:
        self.embedder = embedder
        self.reranker = reranker

    async def search(
        self, session: AsyncSession, query: str, scope: MemoryScope, limit: int
    ) -> list[MemoryHit]:
        params = _scope_params(scope)
        q_emb = await self.embedder.embed_query(query)
        dense = (
            await session.execute(sql(_DENSE_SQL), {**params, "qe": str(q_emb), "k": limit})
        ).all()
        sparse = (
            await session.execute(sql(_SPARSE_SQL), {**params, "q": query, "k": limit})
        ).all()

        # RRF-слияние dense + sparse
        fused: dict[uuid.UUID, MemoryHit] = {}
        scores: dict[uuid.UUID, float] = {}
        for rows in (dense, sparse):
            for rank, row in enumerate(rows):
                hit = fused.setdefault(row.id, _row_to_hit(row))
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if not fused:
            return []
        candidates = sorted(fused.values(), key=lambda h: -scores[h.id])[
            : settings.memory_rerank_top_k
        ]
        for h in candidates:
            h.fused = scores[h.id]

        # reranker — финальная релевантность (gate сравнивает с порогом)
        try:
            rr = await self.reranker.rerank(query, [h.content for h in candidates])
            for h, s in zip(candidates, rr, strict=True):
                h.rerank = s
            candidates.sort(key=lambda h: -h.rerank)
        except Exception as exc:  # reranker недоступен → RRF-порядок, rerank=fused
            logger.warning("memory reranker недоступен (%s) — RRF-порядок", exc)
            for h in candidates:
                h.rerank = h.fused
        return candidates[:limit]

    async def add_or_update(
        self,
        session: AsyncSession,
        *,
        scope_obj: MemoryScope,
        scope: str,
        kind: str,
        content: str,
        structured: dict[str, Any] | None = None,
        sensitivity: str = "normal",
        importance: float = 0.5,
        confidence: float = 0.7,
        source_event_ids: list[uuid.UUID] | None = None,
        fingerprint: str | None = None,
        valid_from: datetime | None = None,
        supersedes: uuid.UUID | None = None,
    ) -> MemoryItem:
        """Создать item (или обновить активный с тем же fingerprint). Эмбеддит
        content. Без commit — коммитит вызывающий сервис."""
        emb = await self._embed(content)
        existing = None
        if fingerprint:
            existing = (
                await session.execute(
                    sql(
                        "SELECT id FROM memory_items WHERE tenant_id = CAST(:t AS uuid)"
                        " AND user_id = :u AND scope = :s AND kind = :k AND fingerprint = :f"
                        " AND status = 'active' AND deleted_at IS NULL LIMIT 1"
                    ),
                    {
                        "t": str(scope_obj.tenant_id),
                        "u": scope_obj.user_id,
                        "s": scope,
                        "k": kind,
                        "f": fingerprint,
                    },
                )
            ).scalar_one_or_none()
        if existing is not None:
            item = await session.get(MemoryItem, existing)
            item.content = content
            item.structured = structured
            item.embedding = emb
            item.confidence = confidence
            item.importance = importance
            item.sensitivity = sensitivity
            return item

        item = MemoryItem(
            tenant_id=scope_obj.tenant_id,
            user_id=scope_obj.user_id,
            project_id=scope_obj.project_id if scope == "project" else None,
            document_id=scope_obj.document_id if scope == "document" else None,
            thread_id=scope_obj.thread_id if scope == "thread" else None,
            scope=scope,
            kind=kind,
            content=content,
            structured=structured,
            source_event_ids=source_event_ids or [],
            confidence=confidence,
            importance=importance,
            sensitivity=sensitivity,
            valid_from=valid_from,
            supersedes=supersedes,
            fingerprint=fingerprint,
            memory_provider="internal",
            embedding=emb,
        )
        session.add(item)
        return item

    async def delete(self, session: AsyncSession, item_id: uuid.UUID) -> None:
        item = await session.get(MemoryItem, item_id)
        if item is not None:
            from datetime import datetime as _dt

            item.status = "deleted"
            item.deleted_at = _dt.now(UTC)

    async def _embed(self, content: str) -> list[float] | None:
        try:
            out = await self.embedder.embed([content])
            return out[0] if out else None
        except Exception as exc:
            logger.warning("memory embed недоступен (%s) — item без вектора", exc)
            return None

    async def healthcheck(self) -> dict[str, Any]:
        try:
            await self.embedder.embed_query("healthcheck")
            return {"provider": "internal", "embed": "ok"}
        except Exception as exc:
            return {"provider": "internal", "embed": f"error: {exc}"}
