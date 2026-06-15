"""API браузерного расширения (roadmap § 6): перевод выделения и батч DOM-узлов.

Разметка страницы на сервер не уезжает — только тексты узлов с их id
(паттерн Immersive Translate, § 3.3.C).
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from rag_app.api.auth import require_user
from rag_app.config import settings
from rag_app.db.models import GlossaryTerm
from rag_app.llm.client import SegmentContext, pick_glossary_terms

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["widget"], dependencies=[require_user])


class SelectionIn(BaseModel):
    text: str = Field(min_length=1)
    target_lang: str = "ru"


async def _load_glossary(request: Request) -> list[tuple[str, str]]:
    """Все термины глоссария (для терминологической интервенции HY-MT)."""
    async with request.app.state.sessionmaker() as db:
        rows = (await db.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all()
    return [(r.en_term, r.ru_term) for r in rows]


@router.post("/selection/translate")
async def selection_translate(request: Request, body: SelectionIn) -> dict:
    text = body.text.strip()
    if len(text) > settings.selection_max_chars:
        raise HTTPException(413, f"выделение длиннее {settings.selection_max_chars} символов")
    # глоссарий применяется и в быстром контуре — через terminology-intervention HY-MT
    glossary = pick_glossary_terms(text, await _load_glossary(request))
    t0 = time.monotonic()
    translated, engine = await request.app.state.fast_translator.translate(
        text, body.target_lang, glossary=glossary
    )
    return {
        "text": translated,
        "engine": engine,
        "ms": int((time.monotonic() - t0) * 1000),
    }


class FragmentIn(BaseModel):
    text: str = Field(min_length=1)


@router.post("/translate/fragment")
async def translate_fragment(request: Request, body: FragmentIn) -> dict:
    """Перевод произвольного фрагмента документ-качеством (Qwen3 + глоссарий) —
    режим «выделенный фрагмент» из ТЗ §4.2 для веб-приложения."""
    text = body.text.strip()
    if len(text) > 8000:
        raise HTTPException(413, "фрагмент длиннее 8000 символов — переведите документ целиком")
    async with request.app.state.sessionmaker() as db:
        rows = (await db.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all()
    terms = pick_glossary_terms(text, [(r.en_term, r.ru_term) for r in rows])
    t0 = time.monotonic()
    out = await request.app.state.translator.translate(text, SegmentContext(glossary=terms))
    return {"text": out, "engine": settings.llm_model, "ms": int((time.monotonic() - t0) * 1000)}


class NodeIn(BaseModel):
    id: str = Field(max_length=32)
    text: str


class WebTranslateIn(BaseModel):
    items: list[NodeIn]
    target_lang: str = "ru"


@router.post("/web/translate")
async def web_translate(request: Request, body: WebTranslateIn) -> dict:
    if len(body.items) > settings.web_translate_max_items:
        raise HTTPException(413, f"не больше {settings.web_translate_max_items} узлов за запрос")
    t0 = time.monotonic()
    translator = request.app.state.fast_translator
    # глоссарий грузим один раз на запрос, матчим под каждый узел
    all_terms = await _load_glossary(request)
    sem = asyncio.Semaphore(settings.web_translate_concurrency)
    engines: set[str] = set()

    async def work(item: NodeIn) -> dict:
        async with sem:
            try:
                terms = pick_glossary_terms(item.text, all_terms)
                translated, engine = await translator.translate(
                    item.text, body.target_lang, glossary=terms
                )
                engines.add(engine)
                return {"id": item.id, "text": translated}
            except Exception as exc:
                logger.error("web_translate узел %s: %s", item.id, exc)
                return {"id": item.id, "text": item.text}  # узел остаётся как есть

    results = await asyncio.gather(*(work(i) for i in body.items))
    return {
        "items": results,
        "engine": ", ".join(sorted(engines - {"none"})) or "none",
        "ms": int((time.monotonic() - t0) * 1000),
    }
