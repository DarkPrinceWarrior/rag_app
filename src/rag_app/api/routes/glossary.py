"""Глоссарий: утверждённая терминология EN→RU (roadmap § 3.4 п.1)."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag_app.db.models import GlossaryTerm

router = APIRouter(prefix="/api/glossary", tags=["glossary"])


class TermIn(BaseModel):
    en_term: str = Field(min_length=1, max_length=256)
    ru_term: str = Field(min_length=1, max_length=256)
    domain: str | None = None


class TermOut(TermIn):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime


@router.get("", response_model=list[TermOut])
async def list_terms(request: Request) -> list[TermOut]:
    async with request.app.state.sessionmaker() as session:
        terms = (
            (await session.execute(select(GlossaryTerm).order_by(GlossaryTerm.en_term))).scalars().all()
        )
    return [TermOut.model_validate(t) for t in terms]


@router.put("", response_model=TermOut)
async def upsert_term(request: Request, body: TermIn) -> TermOut:
    """Создание/обновление по en_term (без учёта регистра не сводим — термин точный)."""
    async with request.app.state.sessionmaker() as session:
        stmt = (
            pg_insert(GlossaryTerm)
            .values(
                id=uuid.uuid4(),
                en_term=body.en_term.strip(),
                ru_term=body.ru_term.strip(),
                domain=body.domain,
            )
            .on_conflict_do_update(
                index_elements=[GlossaryTerm.en_term],
                set_={"ru_term": body.ru_term.strip(), "domain": body.domain},
            )
            .returning(GlossaryTerm)
        )
        term = (await session.execute(stmt)).scalar_one()
        await session.commit()
        return TermOut.model_validate(term)


@router.delete("/{term_id}", status_code=204)
async def delete_term(request: Request, term_id: uuid.UUID) -> None:
    async with request.app.state.sessionmaker() as session:
        result = await session.execute(delete(GlossaryTerm).where(GlossaryTerm.id == term_id))
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "термин не найден")
