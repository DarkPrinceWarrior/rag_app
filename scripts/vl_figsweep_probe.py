"""Проба VL-обхода страниц на ОДНОМ документе (deeplearningbook) — без записи в БД.

Рендерит каждую страницу → Qwen3.5-35B-A3B (:8006) с промптом
«опиши, только если есть рисунок/схема/график, иначе EMPTY» → печатает результат.
Цель: оценить качество описаний диаграмм перед тем, как вшивать обход в
describe_images и переиндексировать. Запуск: uv run python scripts/vl_figsweep_probe.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid

import pypdfium2 as pdfium
from openai import AsyncOpenAI
from PIL import Image
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Document
from rag_app.storage.s3 import Storage

FILENAME_LIKE = "deeplearningbook"

# Промпт-кандидат для продакшна: описывай рисунок/схему, иначе EMPTY.
_SYSTEM = (
    "Ты — инженер технической документации. Тебе дают изображение СТРАНИЦЫ "
    "технического документа. Твоя задача — описать ТОЛЬКО визуальные объекты "
    "(рисунок, схему, диаграмму, граф, график, иллюстрацию, фото), если они есть."
)
_PROMPT = (
    "Если на странице есть рисунок / схема / диаграмма / граф / график / иллюстрация — "
    "кратко опиши ПО-РУССКИ только этот объект: что изображено, ключевые элементы, "
    "связи, инженерный/смысловой смысл; если есть подпись (Figure N / Рис. N) — укажи номер. "
    "Если страница содержит ТОЛЬКО текст, заголовки, формулы и таблицы без рисунков — "
    "ответь РОВНО одним словом: EMPTY"
)


def _render(pdf_bytes: bytes, max_pages: int, scale: float, max_side: int) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(min(len(pdf), max_pages)):
            img = pdf[i].render(scale=scale).to_pil().convert("RGB")
            img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            out.append((i, buf.getvalue()))
    finally:
        pdf.close()
    return out


async def describe(client: AsyncOpenAI, jpg: bytes) -> str:
    b64 = base64.b64encode(jpg).decode("ascii")
    resp = await client.chat.completions.create(
        model=settings.vl_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        temperature=0.2,
        max_tokens=settings.vl_max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    st = Storage()
    async with sm() as s:
        doc = (
            await s.execute(select(Document).where(Document.filename.like(f"%{FILENAME_LIKE}%")))
        ).scalars().first()
    if not doc:
        raise SystemExit(f"нет документа like {FILENAME_LIKE!r}")
    print(f"doc {doc.id} {doc.filename} kind={doc.kind} pages={doc.page_count}")
    data = await st.get_bytes(settings.bucket_originals, doc.s3_key_original)
    pages = await asyncio.to_thread(_render, data, 50, settings.vl_render_scale, settings.vl_max_side)
    print(f"страниц к обходу: {len(pages)}  (scale={settings.vl_render_scale}, side={settings.vl_max_side})\n")

    client = AsyncOpenAI(base_url=settings.vl_base_url, api_key=settings.llm_api_key, timeout=180.0)
    fig, empty = 0, 0
    t0 = time.monotonic()
    for pidx, jpg in pages:
        try:
            desc = await describe(client, jpg)
        except Exception as exc:
            print(f"--- стр.{pidx + 1}: ОШИБКА {exc}")
            continue
        if desc.upper().strip(". ") == "EMPTY" or len(desc) < 8:
            empty += 1
            print(f"--- стр.{pidx + 1}: EMPTY")
        else:
            fig += 1
            print(f"=== стр.{pidx + 1} (рисунок) ===\n{desc}\n")
    dt = time.monotonic() - t0
    print(f"\nИТОГО: {fig} с рисунком, {empty} EMPTY, {len(pages)} страниц, {dt:.0f} с ({dt / max(len(pages),1):.1f} с/стр)")


if __name__ == "__main__":
    asyncio.run(main())
