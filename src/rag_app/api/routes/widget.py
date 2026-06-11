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

from rag_app.api.auth import require_user
from rag_app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["widget"], dependencies=[require_user])


class SelectionIn(BaseModel):
    text: str = Field(min_length=1)
    target_lang: str = "ru"


@router.post("/selection/translate")
async def selection_translate(request: Request, body: SelectionIn) -> dict:
    text = body.text.strip()
    if len(text) > settings.selection_max_chars:
        raise HTTPException(413, f"выделение длиннее {settings.selection_max_chars} символов")
    t0 = time.monotonic()
    translated, engine = await request.app.state.fast_translator.translate(text, body.target_lang)
    return {
        "text": translated,
        "engine": engine,
        "ms": int((time.monotonic() - t0) * 1000),
    }


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
    sem = asyncio.Semaphore(settings.web_translate_concurrency)
    engines: set[str] = set()

    async def work(item: NodeIn) -> dict:
        async with sem:
            try:
                translated, engine = await translator.translate(item.text, body.target_lang)
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
