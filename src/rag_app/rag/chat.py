"""RAG-чат с документом/библиотекой (roadmap § 5): single-hop hybrid+rerank,
стрим токенов, обязательные цитаты [n].

Agentic-уровень (§ 5 п.7, multi-hop tool-цикл) — следующая итерация этапа;
все стоп-условия дизайна будут там.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
Ты — ассистент по корпоративной технической документации (нефтегаз, строительство, договоры).
Отвечай на русском языке, точно и по делу.

Правила:
1. Отвечай ТОЛЬКО на основании приведённых фрагментов документов. Если ответа
   в них нет — прямо скажи «В документах ответа не нашлось» и ничего не выдумывай.
2. После каждого утверждения ставь ссылку на фрагмент в виде [n], где n — номер
   фрагмента. Ссылки обязательны.
3. Числа, единицы измерения и обозначения стандартов переноси без изменений.
4. Если фрагменты противоречат друг другу — отметь это явно."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for n, c in enumerate(chunks, 1):
        header = f"[{n}] {c.filename}"
        if c.heading_path:
            header += f" · {c.heading_path}"
        if c.page_start is not None:
            pages = f"стр. {c.page_start + 1}" + (
                f"–{c.page_end + 1}" if c.page_end is not None and c.page_end != c.page_start else ""
            )
            header += f" · {pages}"
        body = c.text_ru or c.text_en
        parts.append(f"{header}\n{body[:3000]}")
    return "\n\n---\n\n".join(parts)


_CITATION = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Цитаты из ответа → метаданные чанков (страница, bbox для подсветки)."""
    seen: list[dict[str, Any]] = []
    used: set[int] = set()
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if n in used or not (1 <= n <= len(chunks)):
            continue
        used.add(n)
        c = chunks[n - 1]
        seen.append(
            {
                "n": n,
                "chunk_id": str(c.id),
                "document_id": str(c.document_id),
                "filename": c.filename,
                "heading_path": c.heading_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "bboxes": (c.meta or {}).get("bboxes", [])[:20],
                "segment_ids": (c.meta or {}).get("segment_ids", [])[:30],
            }
        )
    return sorted(seen, key=lambda x: x["n"])


class ChatEngine:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=300.0
        )

    async def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        context_block = build_context_block(chunks)
        messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        messages.extend(history[-settings.rag_history_messages :])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Фрагменты документов:\n\n{context_block}\n\n"
                    f"Вопрос: {question}\n\n"
                    "Ответь по правилам (цитаты [n] обязательны)."
                ),
            }
        )
        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.2,
            top_p=0.8,
            max_tokens=2048,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def make_session_title(question: str) -> str:
    title = " ".join(question.split())
    return title[:77] + "…" if len(title) > 78 else title or "Новый чат"


def new_id() -> uuid.UUID:
    return uuid.uuid4()
