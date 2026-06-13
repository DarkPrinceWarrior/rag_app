"""Инструменты agentic-RAG (§ 5 п.7): retrieval как tool call.

Каждый фрагмент, который агент достаёт, попадает в evidence-пул
(ref = первые 8 символов uuid). После цикла из пула собирается финальный
контекст ответа. Наблюдения компактны — токен-бюджет цикла ограничен
(стоп-условия в agent.py)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.models import ChatMessage, Chunk, Document
from rag_app.rag.retrieve import RetrievedChunk, Retriever


def _ref(chunk_id: uuid.UUID) -> str:
    return str(chunk_id)[:8]


def _pages(c: RetrievedChunk) -> str:
    if c.page_start is None:
        return ""
    out = f"стр. {c.page_start + 1}"
    if c.page_end is not None and c.page_end != c.page_start:
        out += f"–{c.page_end + 1}"
    return out


class AgentTools:
    """Набор инструментов агента поверх одной сессии чата. Держит evidence-пул
    (ref → чанк) — из него потом собирается контекст финального ответа."""

    def __init__(
        self,
        sessionmaker: Any,
        retriever: Retriever,
        *,
        document_id: uuid.UUID | None,
        folder_id: uuid.UUID | None,
        session_id: uuid.UUID,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.retriever = retriever
        self.document_id = document_id
        self.folder_id = folder_id
        self.session_id = session_id
        self.evidence: dict[str, RetrievedChunk] = {}

    def _register(self, chunks: list[RetrievedChunk]) -> None:
        for c in chunks:
            self.evidence.setdefault(_ref(c.id), c)

    def _fmt(self, c: RetrievedChunk, body_len: int) -> str:
        head = f"[{_ref(c.id)}] {c.filename}"
        if c.heading_path:
            head += f" · {c.heading_path}"
        pages = _pages(c)
        if pages:
            head += f" · {pages}"
        body = (c.text_ru or c.text_en or "").strip().replace("\n", " ")
        return f"{head}\n{body[:body_len]}"

    async def search_chunks(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "search_chunks: пустой запрос."
        async with self.sessionmaker() as db:
            chunks = await self.retriever.retrieve(
                db, query, document_id=self.document_id, folder_id=self.folder_id,
                top_k=settings.agent_search_top_k,
            )
        if not chunks:
            return f"search_chunks('{query[:80]}'): ничего не найдено."
        self._register(chunks)
        body = "\n---\n".join(self._fmt(c, 220) for c in chunks)
        return f"search_chunks('{query[:80]}'): {len(chunks)} фрагм.\n{body}"

    async def get_section(self, ref: str) -> str:
        c = self.evidence.get((ref or "").strip()[:8])
        if c is None:
            return f"get_section('{ref}'): нет такого фрагмента (ref берётся из search_chunks)."
        return f"get_section('{ref}'):\n{self._fmt(c, 2800)}"

    async def get_tables(self, document_id: str | None = None) -> str:
        doc = self.document_id
        if document_id:
            try:
                doc = uuid.UUID(document_id)
            except ValueError:
                pass
        if doc is None:
            return "get_tables: нужен document_id (или открой конкретный документ в чате)."
        async with self.sessionmaker() as db:
            rows = (
                await db.execute(
                    select(Chunk, Document.filename)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(Chunk.document_id == doc, Chunk.kind == "table")
                    .order_by(Chunk.idx)
                )
            ).all()
        if not rows:
            return "get_tables: таблиц в документе не найдено."
        chunks = [
            RetrievedChunk(
                id=ch.id, document_id=ch.document_id, filename=fn, heading_path=ch.heading_path,
                kind=ch.kind, page_start=ch.page_start, page_end=ch.page_end,
                text_en=ch.text_en, text_ru=ch.text_ru, meta=ch.meta,
            )
            for ch, fn in rows
        ]
        self._register(chunks)
        body = "\n---\n".join(self._fmt(c, 400) for c in chunks)
        return f"get_tables: {len(chunks)} табл.\n{body}"

    async def get_chat_history(self) -> str:
        async with self.sessionmaker() as db:
            rows = (
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.session_id == self.session_id)
                        .order_by(ChatMessage.created_at)
                    )
                )
                .scalars()
                .all()
            )
        prior = rows[:-1][-10:]  # без только что записанного вопроса
        if not prior:
            return "get_chat_history: предыдущих реплик в этой сессии нет."
        return "get_chat_history:\n" + "\n".join(f"{m.role}: {m.content[:300]}" for m in prior)

    async def dispatch(self, action: str, *, query: str = "", ref: str = "", document_id: str = "") -> str:
        if action == "search_chunks":
            return await self.search_chunks(query)
        if action == "get_section":
            return await self.get_section(ref)
        if action == "get_tables":
            return await self.get_tables(document_id or None)
        if action == "get_chat_history":
            return await self.get_chat_history()
        return f"неизвестный инструмент: {action}"
