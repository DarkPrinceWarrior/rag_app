"""API слоя памяти (docs/MEMORY_rev4_mem0_articles.md §8): ручной контроль
пользователя над тем, что приложение о нём помнит.

Этап 1 — CRUD над `memory_items` (просмотр/добавление/правка/удаление).
Очередь кандидатов, purge и health провайдера — Этапы 2–4.
Все операции owner-scoped: пользователь видит и меняет только свои items
(admin — любые); tenant_id — константа (single-org).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from rag_app.api.audit import audit
from rag_app.api.auth import User, require_user
from rag_app.db.models import MemoryCandidate, MemoryItem
from rag_app.rag.memory.consolidate import promote_candidate, reject_candidate
from rag_app.rag.memory.rls import apply_scope_guc

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[require_user])

_SCOPES = {"user", "project", "document", "thread", "org"}
_KINDS = {"preference", "fact", "glossary", "rule", "task", "correction", "summary"}
_SENS = {"normal", "sensitive", "secret"}


class MemoryIn(BaseModel):
    scope: str = "user"
    kind: str = "fact"
    content: str = Field(min_length=1, max_length=8000)
    project_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    thread_id: uuid.UUID | None = None
    sensitivity: str = "normal"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryPatch(BaseModel):
    content: str | None = Field(default=None, max_length=8000)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity: str | None = None


def _item_out(it: MemoryItem) -> dict:
    return {
        "id": str(it.id),
        "scope": it.scope,
        "kind": it.kind,
        "content": it.content,
        "sensitivity": it.sensitivity,
        "importance": it.importance,
        "confidence": it.confidence,
        "project_id": str(it.project_id) if it.project_id else None,
        "document_id": str(it.document_id) if it.document_id else None,
        "thread_id": str(it.thread_id) if it.thread_id else None,
        "memory_provider": it.memory_provider,
        "created_at": it.created_at.isoformat() if it.created_at else None,
        "updated_at": it.updated_at.isoformat() if it.updated_at else None,
    }


def _owner_ok(it: MemoryItem | MemoryCandidate, user: User) -> bool:
    return user.is_admin or it.user_id == user.sub


def _cand_out(c: MemoryCandidate) -> dict:
    return {
        "id": str(c.id),
        "action": c.action,
        "status": c.status,
        "confidence": c.confidence,
        "proposed": c.proposed,
        "rationale": c.rationale,
        "created_at": c.created_at.isoformat(),
    }


@router.get("")
async def list_memory(
    request: Request,
    scope: str | None = None,
    project_id: uuid.UUID | None = None,
    q: str | None = None,
) -> list[dict]:
    user: User = request.state.user
    memory = request.app.state.memory
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        items = await memory.list_items(
            db,
            user.sub,
            is_admin=user.is_admin,
            scope_filter=scope,
            project_id=project_id,
            q=q,
        )
    return [_item_out(it) for it in items]


@router.post("", status_code=201)
async def create_memory(request: Request, body: MemoryIn) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    if body.scope not in _SCOPES:
        raise HTTPException(422, f"scope ∈ {sorted(_SCOPES)}")
    if body.kind not in _KINDS:
        raise HTTPException(422, f"kind ∈ {sorted(_KINDS)}")
    if body.sensitivity not in _SENS:
        raise HTTPException(422, f"sensitivity ∈ {sorted(_SENS)}")
    mem_scope = memory.scope_for(
        user.sub,
        project_id=body.project_id,
        document_id=body.document_id,
        thread_id=body.thread_id,
    )
    async with request.app.state.sessionmaker() as db:
        item = await memory.add_manual(
            db,
            mem_scope,
            scope_kind=body.scope,
            kind=body.kind,
            content=body.content,
            sensitivity=body.sensitivity,
            importance=body.importance,
            actor="admin" if user.is_admin else "user",
        )
        await db.commit()
        out = _item_out(item)
    await audit(request, "memory_create", "memory_item", out["id"], {"kind": body.kind})
    return out


@router.patch("/{item_id}")
async def update_memory(request: Request, item_id: uuid.UUID, body: MemoryPatch) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    if body.sensitivity is not None and body.sensitivity not in _SENS:
        raise HTTPException(422, f"sensitivity ∈ {sorted(_SENS)}")
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        item = await db.get(MemoryItem, item_id)
        if item is None or item.status != "active" or not _owner_ok(item, user):
            raise HTTPException(404, "запись памяти не найдена")
        await memory.update_item(
            db,
            item,
            content=body.content,
            importance=body.importance,
            sensitivity=body.sensitivity,
            actor="admin" if user.is_admin else "user",
        )
        await db.commit()
        out = _item_out(item)
    await audit(request, "memory_update", "memory_item", str(item_id), None)
    return out


@router.delete("/{item_id}", status_code=204)
async def delete_memory(request: Request, item_id: uuid.UUID) -> None:
    user: User = request.state.user
    memory = request.app.state.memory
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        item = await db.get(MemoryItem, item_id)
        if item is None or item.status != "active" or not _owner_ok(item, user):
            raise HTTPException(404, "запись памяти не найдена")
        await memory.delete_item(db, item, actor="admin" if user.is_admin else "user")
        await db.commit()
    await audit(request, "memory_delete", "memory_item", str(item_id), None)


# --- 152-ФЗ: экспорт и удаление памяти пользователя (§8, Этап 3) -------------


class PurgeIn(BaseModel):
    user_id: str | None = None  # None → сам пользователь; admin может указать чужого


@router.post("/purge")
async def purge_memory(request: Request, body: PurgeIn) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    target = body.user_id or user.sub
    if target != user.sub and not user.is_admin:
        raise HTTPException(403, "удалять чужую память может только администратор")
    async with request.app.state.sessionmaker() as db:
        counts = await memory.purge_user(db, target)
        await db.commit()
    await audit(request, "memory_purge", "memory_user", target, counts)
    return {"purged": target, **counts}


@router.get("/export")
async def export_memory(request: Request, user_id: str | None = None) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    target = user_id or user.sub
    if target != user.sub and not user.is_admin:
        raise HTTPException(403, "экспортировать чужую память может только администратор")
    async with request.app.state.sessionmaker() as db:
        data = await memory.export_user(db, target)
    await audit(request, "memory_export", "memory_user", target, None)
    return data


# --- Очередь кандидатов автоэкстрактора (§8, Этап 2) ------------------------


@router.get("/candidates")
async def list_candidates(request: Request, status: str = "pending") -> list[dict]:
    user: User = request.state.user
    memory = request.app.state.memory
    stmt = (
        select(MemoryCandidate)
        .where(MemoryCandidate.tenant_id == memory.tenant_id, MemoryCandidate.status == status)
        .order_by(MemoryCandidate.created_at.desc())
        .limit(200)
    )
    if not user.is_admin:
        stmt = stmt.where(MemoryCandidate.user_id == user.sub)
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        rows = (await db.execute(stmt)).scalars().all()
    return [_cand_out(c) for c in rows]


@router.post("/candidates/{cand_id}/accept", status_code=201)
async def accept_candidate(request: Request, cand_id: uuid.UUID) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    actor = "admin" if user.is_admin else "user"
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        cand = await db.get(MemoryCandidate, cand_id)
        if cand is None or cand.status != "pending" or not _owner_ok(cand, user):
            raise HTTPException(404, "кандидат не найден")
        item = await promote_candidate(db, memory, cand, actor=actor)
        await db.commit()
        out = {"candidate": _cand_out(cand), "item_id": str(item.id) if item else None}
    await audit(request, "memory_accept_candidate", "memory_candidate", str(cand_id), None)
    return out


@router.post("/candidates/{cand_id}/reject", status_code=200)
async def reject_candidate_route(request: Request, cand_id: uuid.UUID) -> dict:
    user: User = request.state.user
    memory = request.app.state.memory
    actor = "admin" if user.is_admin else "user"
    async with request.app.state.sessionmaker() as db:
        await apply_scope_guc(db, memory.scope_for(user.sub))
        cand = await db.get(MemoryCandidate, cand_id)
        if cand is None or cand.status != "pending" or not _owner_ok(cand, user):
            raise HTTPException(404, "кандидат не найден")
        await reject_candidate(db, cand, actor=actor)
        await db.commit()
        out = _cand_out(cand)
    await audit(request, "memory_reject_candidate", "memory_candidate", str(cand_id), None)
    return out
