"""Управление кандидатами translation memory: утверждение и отзыв."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from rag_app.api.audit import audit
from rag_app.api.auth import User, require_user
from rag_app.db.models import TranslationMemory

router = APIRouter(
    prefix="/api/translation-memory",
    tags=["translation-memory"],
    dependencies=[require_user],
)


class TranslationMemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_text: str
    approved_translation: str
    source_lang: str
    target_lang: str
    domain: str
    project: str | None
    folder_id: uuid.UUID | None
    editor_name: str | None
    status: str
    approved_at: datetime | None
    revoked_at: datetime | None
    revocation_reason: str | None
    created_at: datetime


class RevokeIn(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


def _owned(entry: TranslationMemory | None, user: User) -> TranslationMemory:
    if entry is None or (not user.is_admin and entry.owner_sub != user.sub):
        raise HTTPException(404, "запись памяти не найдена")
    return entry


@router.get("", response_model=list[TranslationMemoryOut])
async def list_translation_memory(
    request: Request,
    status: Annotated[Literal["candidate", "approved", "revoked"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[TranslationMemoryOut]:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as session:
        stmt = select(TranslationMemory).order_by(TranslationMemory.created_at.desc()).limit(limit)
        if not user.is_admin:
            stmt = stmt.where(TranslationMemory.owner_sub == user.sub)
        if status is not None:
            stmt = stmt.where(TranslationMemory.status == status)
        rows = (await session.execute(stmt)).scalars().all()
    return [TranslationMemoryOut.model_validate(row) for row in rows]


@router.post("/{entry_id}/approve", response_model=TranslationMemoryOut)
async def approve_translation_memory(
    request: Request,
    entry_id: uuid.UUID,
) -> TranslationMemoryOut:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as session:
        entry = _owned(await session.get(TranslationMemory, entry_id), user)
        if entry.status != "candidate":
            raise HTTPException(409, "утвердить можно только кандидата")
        source_text = entry.source_text

    vector = (await request.app.state.retriever.embedder.embed([source_text]))[0]
    now = datetime.now(UTC)
    async with request.app.state.sessionmaker() as session:
        entry = _owned(await session.get(TranslationMemory, entry_id), user)
        if entry.status != "candidate":
            raise HTTPException(409, "статус кандидата уже изменился")
        await session.execute(
            update(TranslationMemory)
            .where(
                TranslationMemory.id != entry.id,
                TranslationMemory.owner_sub == entry.owner_sub,
                TranslationMemory.folder_id.is_not_distinct_from(entry.folder_id),
                TranslationMemory.project.is_not_distinct_from(entry.project),
                TranslationMemory.domain == entry.domain,
                TranslationMemory.source_lang == entry.source_lang,
                TranslationMemory.target_lang == entry.target_lang,
                TranslationMemory.source_hash == entry.source_hash,
                TranslationMemory.status == "approved",
                TranslationMemory.revoked_at.is_(None),
            )
            .values(
                status="revoked",
                revoked_by_sub=user.sub,
                revoked_at=now,
                revocation_reason="заменена более новой утверждённой правкой",
                updated_at=now,
            )
        )
        entry.status = "approved"
        entry.source_embedding = vector
        entry.approved_by_sub = user.sub
        entry.approved_at = now
        entry.revoked_by_sub = None
        entry.revoked_at = None
        entry.revocation_reason = None
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(409, "конкурирующее утверждение уже заняло exact-запись") from None
        await session.refresh(entry)
        result = TranslationMemoryOut.model_validate(entry)
    await audit(request, "translation_memory_approve", "translation_memory", str(entry_id))
    return result


@router.post("/{entry_id}/revoke", response_model=TranslationMemoryOut)
async def revoke_translation_memory(
    request: Request,
    entry_id: uuid.UUID,
    body: RevokeIn,
) -> TranslationMemoryOut:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as session:
        entry = _owned(await session.get(TranslationMemory, entry_id), user)
        if entry.status == "revoked":
            raise HTTPException(409, "запись уже отозвана")
        entry.status = "revoked"
        entry.revoked_by_sub = user.sub
        entry.revoked_at = datetime.now(UTC)
        entry.revocation_reason = body.reason.strip()
        await session.commit()
        await session.refresh(entry)
        result = TranslationMemoryOut.model_validate(entry)
    await audit(
        request,
        "translation_memory_revoke",
        "translation_memory",
        str(entry_id),
        {"reason": body.reason.strip()},
    )
    return result
