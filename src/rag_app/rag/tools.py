"""Инструменты agentic-RAG (§ 5 п.7): retrieval как tool call.

Каждый фрагмент, который агент достаёт, попадает в evidence-пул
(ref = первые 8 символов uuid). После цикла из пула собирается финальный
контекст ответа. Наблюдения компактны — токен-бюджет цикла ограничен
(стоп-условия в agent.py)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from sqlalchemy import or_, select

from rag_app.config import settings
from rag_app.db.models import ChatMessage, Chunk, Document, DocumentStatus
from rag_app.rag.retrieve import RetrievedChunk, Retriever

# Номер рисунка/таблицы из запроса: "9.1", "3", "12-4" (допускаем точку/дефис/двоеточие
# как разделитель составного номера, как в "Figure 9.1" / "Table 2.3").
_FIGNUM_RE = re.compile(r"\d+(?:[.\-]\d+)*")
# Ключевые слова перед номером — для подсказки агенту, но матчим в основном по номеру.
_TABLE_HINTS = ("table", "табл")  # рисунок vs таблица — слабый сигнал, не жёсткий фильтр


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
        owner_sub: str | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.retriever = retriever
        self.document_id = document_id
        self.document_ids = document_ids or None
        self.folder_id = folder_id
        self.session_id = session_id
        self.owner_sub = owner_sub
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
                document_ids=self.document_ids, top_k=settings.agent_search_top_k,
                owner_sub=self.owner_sub,  # RBAC §4.7.1: поиск агента — только свои документы
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

    async def get_chapter(self, designation: str, document_id: str | None = None) -> str:
        """Раздел/глава ЦЕЛИКОМ по номеру из заголовка (heading_path), а не один
        фрагмент (ТЗ §4.4.3c, «перескажи главу 3»). Собирает все фрагменты раздела
        и его подразделов по совпадению номера в пути заголовков."""
        m = _FIGNUM_RE.search(designation or "")
        if not m:
            return (
                f"get_chapter('{(designation or '')[:60]}'): не разобрал номер раздела "
                "(ожидается «3», «4.2» и т.п.)."
            )
        num = m.group(0)
        # «пункт 5» = РАЗДЕЛ ВЕРХНЕГО УРОВНЯ «5. …» (heading_path начинается с номера),
        # а не вложенный «2. … → 5. …». Поэтому сначала ищем верхний уровень; если
        # его нет (составной номер «4.2» лежит внутри пути) — номер где угодно в пути:
        # слева не цифра/точка/дефис (чтобы «3» не цеплял «13»), справа — не цифра.
        top_pat = rf"^\s*{re.escape(num)}[.\s]"
        broad_pat = rf"(^|[^0-9.\-]){re.escape(num)}([^0-9]|$)"
        doc = self.document_id
        if document_id:
            try:
                doc = uuid.UUID(document_id)
            except ValueError:
                pass

        def _scoped(pat: str):
            stmt = (
                select(Chunk, Document.filename)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.heading_path.op("~")(pat))
            )
            if doc is not None:
                stmt = stmt.where(Chunk.document_id == doc)
            elif self.document_ids:
                stmt = stmt.where(Chunk.document_id.in_(self.document_ids))
            elif self.folder_id is not None:
                stmt = stmt.where(Document.folder_id == self.folder_id)
            if self.owner_sub is not None:  # RBAC §4.7.1: раздел только своих документов
                stmt = stmt.where(
                    (Document.owner_sub == self.owner_sub) | (Document.owner_sub.is_(None))
                )
            return stmt.order_by(Chunk.idx).limit(settings.agent_max_context_chunks)

        async with self.sessionmaker() as db:
            rows = (await db.execute(_scoped(top_pat))).all()
            if not rows:  # верхнеуровневого «N.» нет — ищем номер где угодно в пути
                rows = (await db.execute(_scoped(broad_pat))).all()
        if not rows:
            return f"get_chapter('{num}'): раздел {num} не найден (поиск по заголовкам)."
        chunks = [
            RetrievedChunk(
                id=ch.id, document_id=ch.document_id, filename=fn, heading_path=ch.heading_path,
                kind=ch.kind, page_start=ch.page_start, page_end=ch.page_end,
                text_en=ch.text_en, text_ru=ch.text_ru, meta=ch.meta,
            )
            for ch, fn in rows
        ]
        self._register(chunks)
        body = "\n---\n".join(self._fmt(c, 600) for c in chunks)
        return f"get_chapter('{num}'): {len(chunks)} фрагм. раздела.\n{body}"

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
            stmt = (
                select(Chunk, Document.filename)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.document_id == doc, Chunk.kind == "table")
            )
            if self.owner_sub is not None:  # RBAC §4.7.1: таблицы только своих документов
                stmt = stmt.where(
                    (Document.owner_sub == self.owner_sub) | (Document.owner_sub.is_(None))
                )
            rows = (await db.execute(stmt.order_by(Chunk.idx))).all()
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

    async def find_figure(self, designation: str, document_id: str | None = None) -> str:
        """Найти image-чанк по ЯВНО названному номеру рисунка/таблицы («рис. 9.1»,
        «figure 3», «таблица 2»). «Путь B» vision-on-demand: подпись индексируется
        как chunk kind='image' с meta.img_s3 (кроп). Достаём такой чанк по номеру в
        подписи, кладём в evidence-пул — его кроп уйдёт в мультимодальный контекст,
        даже если обычный retrieval его не поднял."""
        raw = (designation or "").strip()
        m = _FIGNUM_RE.search(raw)
        if not m:
            return (
                f"find_figure('{raw[:60]}'): не разобрал номер рисунка/таблицы "
                "(ожидается что-то вроде «9.1», «3», «таблица 2»)."
            )
        num = m.group(0)
        # Границы вокруг номера: слева не цифра/точка/дефис, справа — не цифра
        # (иначе «9.1» подцепил бы «9.10», «9.11»). \m/\M не работают на «.»,
        # поэтому явные lookaround-классы. Номер экранируем (точка → литерал).
        num_re = re.escape(num)
        pat = rf"(^|[^0-9.\-]){num_re}([^0-9]|$)"

        doc = self.document_id
        if document_id:
            try:
                doc = uuid.UUID(document_id)
            except ValueError:
                pass

        async with self.sessionmaker() as db:
            stmt = (
                select(Chunk, Document.filename)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    Chunk.kind == "image",
                    or_(Chunk.text_en.op("~*")(pat), Chunk.text_ru.op("~*")(pat)),
                )
            )
            if doc is not None:
                stmt = stmt.where(Chunk.document_id == doc)
            elif self.folder_id is not None:
                stmt = stmt.where(Document.folder_id == self.folder_id)
            elif self.owner_sub is not None:  # RBAC: свои + dev-документы (owner NULL)
                stmt = stmt.where(
                    (Document.owner_sub == self.owner_sub) | (Document.owner_sub.is_(None))
                )
            rows = (await db.execute(stmt.order_by(Chunk.idx).limit(8))).all()

        if not rows:
            scope = "в этом документе" if doc is not None else "в доступных документах"
            return f"find_figure('{num}'): рисунок/таблица с номером {num} не найдены {scope}."
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
        return (
            f"find_figure('{num}'): {len(chunks)} совпад. (кропы приложены к контексту "
            f"для рассмотрения).\n{body}"
        )

    async def list_documents(self) -> str:
        """Каталог библиотеки: какие документы загружены (а не их содержимое).
        Для вопросов «какие документы есть», «назови документ из библиотеки»."""
        async with self.sessionmaker() as db:
            stmt = select(Document).where(Document.status == DocumentStatus.done)
            # Область чата (набор документов / папка) — каталог должен ей
            # соответствовать, иначе при folder-scope агент перечислит всю библиотеку.
            if self.document_ids:
                stmt = stmt.where(Document.id.in_(self.document_ids))
            elif self.document_id is not None:
                stmt = stmt.where(Document.id == self.document_id)
            elif self.folder_id is not None:
                stmt = stmt.where(Document.folder_id == self.folder_id)
            if self.owner_sub is not None:  # RBAC поверх области: свои + dev (owner NULL)
                stmt = stmt.where(
                    (Document.owner_sub == self.owner_sub) | (Document.owner_sub.is_(None))
                )
            rows = (
                (await db.execute(stmt.order_by(Document.created_at.desc()).limit(200)))
                .scalars()
                .all()
            )
        if not rows:
            return "list_documents: в библиотеке нет готовых документов."
        lines = [
            f"- {d.filename}" + (f" ({d.page_count} стр.)" if d.page_count else "") for d in rows
        ]
        catalog = f"Каталог библиотеки — загруженные документы ({len(rows)}):\n" + "\n".join(lines)
        # Синтетический evidence-чанк: каталог должен попасть в финальный ответ.
        # Без него evidence-пул пуст → fallback-ретрив подменяет каталог
        # содержимым документов (агент видел список, а ответ — нет).
        self._register(
            [
                RetrievedChunk(
                    id=uuid.uuid4(), document_id=uuid.uuid4(), filename="Каталог библиотеки",
                    heading_path="", kind="catalog", page_start=None, page_end=None,
                    text_en="", text_ru=catalog, meta={},
                )
            ]
        )
        return f"list_documents: {catalog}"

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
        if action == "get_chapter":
            return await self.get_chapter(query, document_id or None)
        if action == "get_tables":
            return await self.get_tables(document_id or None)
        if action == "find_figure":
            return await self.find_figure(query, document_id or None)
        if action == "get_chat_history":
            return await self.get_chat_history()
        if action == "list_documents":
            return await self.list_documents()
        return f"неизвестный инструмент: {action}"
