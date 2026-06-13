"""Спец-интенты § 5 п.6: «вытащи спецификации в таблицу».

Structured output (response_format=json_schema на vLLM) превращает запрос +
найденные фрагменты в таблицу {title, columns, rows}; рендер и экспорт XLSX —
в роуте. Источники (чанки) возвращаются рядом для цитат и привязки к /view.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.config import settings
from rag_app.rag.chat import build_context_block
from rag_app.rag.retrieve import Retriever

logger = logging.getLogger(__name__)

_TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["title", "columns", "rows"],
    "additionalProperties": False,
}

_SYSTEM = """\
Ты извлекаешь структурированные данные из технической документации в таблицу.
По запросу пользователя и приведённым фрагментам верни строгий JSON
{title, columns, rows}:
- columns — заголовки столбцов (по-русски);
- rows — массив строк, каждая строка — массив значений в порядке columns;
- бери ТОЛЬКО данные из фрагментов; числа, единицы и обозначения стандартов
  переноси без изменений; если данных нет — пустые rows.
Не добавляй пояснений вне JSON."""


async def extract_table(
    client: AsyncOpenAI,
    retriever: Retriever,
    session: AsyncSession,
    query: str,
    document_id: uuid.UUID | None = None,
    folder_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    chunks = await retriever.retrieve(
        session, query, document_id=document_id, folder_id=folder_id,
        top_k=settings.extract_context_top_k,
    )
    sources = [
        {
            "n": i + 1,
            "document_id": str(c.document_id),
            "filename": c.filename,
            "heading_path": c.heading_path,
            "page": (c.page_start + 1) if c.page_start is not None else None,
            "segment_ids": (c.meta or {}).get("segment_ids", [])[:1],
        }
        for i, c in enumerate(chunks)
    ]
    if not chunks:
        return {"title": query[:80], "columns": [], "rows": [], "sources": []}

    context = build_context_block(chunks)
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Фрагменты:\n\n{context}\n\nЗапрос: {query}"},
        ],
        temperature=0.1,
        max_tokens=2500,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "table", "strict": True, "schema": _TABLE_SCHEMA},
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        logger.warning("extract_table: невалидный JSON")
        data = {}
    return {
        "title": data.get("title") or query[:80],
        "columns": data.get("columns", []),
        "rows": data.get("rows", []),
        "sources": sources,
    }
