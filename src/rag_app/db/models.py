"""Схема БД (этап 3): documents + segments + glossary + chunks(+vector)
+ chat_sessions/messages + folders.

Дальше по roadmap добавятся: users, versions, audit_log. Схемой управляет
alembic (`uv run alembic upgrade head`), create_all не используется.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024  # BGE-M3 dense


class Base(DeclarativeBase):
    pass


class DocumentStatus(enum.StrEnum):
    uploaded = "uploaded"
    parsing = "parsing"
    parsed = "parsed"
    translating = "translating"
    translated = "translated"
    exporting = "exporting"
    done = "done"
    error = "error"


class SegmentKind(enum.StrEnum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    equation = "equation"
    image = "image"


class DocumentKind(enum.StrEnum):
    """Маршрут обработки (roadmap § 3.1)."""

    pdf_text = "pdf_text"
    pdf_scan = "pdf_scan"
    docx = "docx"
    xlsx = "xlsx"
    pptx = "pptx"
    text = "text"  # plain TXT (ТЗ §4.2): без OCR, экспорт только DOCX


# Виды сегментов, которые отправляются на перевод.
TRANSLATABLE_KINDS = (SegmentKind.heading, SegmentKind.paragraph, SegmentKind.table, SegmentKind.image)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"), default=DocumentStatus.uploaded
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)

    # pdf_text|pdf_scan|docx|xlsx|pptx; строка, не SQL-enum — маршруты будут расти
    kind: Mapped[str] = mapped_column(String(16), default=DocumentKind.pdf_text.value)

    # Форс-OCR (восстановление документов с битым ToUnicode-cmap текстового слоя):
    # парсинг через MinerU -m ocr с указанным языком (en|east_slavic|…). Хранится
    # на документе, чтобы переживать retry/reexport.
    parse_force_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_lang: Mapped[str | None] = mapped_column(String(16), default=None)
    # Выбор парсера pdf_text на документе (null → settings.pdf_parser_backend).
    # mineru | dots_mocr | paddle_vl. Переживает retry/reexport.
    parser_backend: Mapped[str | None] = mapped_column(String(16), default=None)

    s3_key_original: Mapped[str] = mapped_column(String(1024))
    s3_key_content_list: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_export_docx: Mapped[str | None] = mapped_column(String(1024), default=None)
    # BabelDOC: PDF с сохранённой вёрсткой (mono — только перевод, dual — EN+RU)
    s3_key_export_pdf: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_export_pdf_dual: Mapped[str | None] = mapped_column(String(1024), default=None)
    # OOXML-ветка: переведённый файл исходного формата (docx/xlsx/pptx)
    s3_key_export_source: Mapped[str | None] = mapped_column(String(1024), default=None)
    # Рендер OOXML в PDF (LibreOffice) для просмотра «как в Microsoft»:
    # оригинал и перевод — в pdf.js-вьювере вместо плоского текста.
    s3_key_view_orig: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_view_ru: Mapped[str | None] = mapped_column(String(1024), default=None)

    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    translated_count: Mapped[int] = mapped_column(Integer, default=0)

    # RAG-индекс (этап 3)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    index_error: Mapped[str | None] = mapped_column(Text, default=None)
    folder_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"), default=None
    )
    # RBAC (этап 5): sub владельца из OIDC-токена; NULL — документы dev-периода
    owner_sub: Mapped[str | None] = mapped_column(String(64), default=None, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    segments: Mapped[list[Segment]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)  # порядок в документе
    page_idx: Mapped[int | None] = mapped_column(Integer, default=None)
    kind: Mapped[SegmentKind] = mapped_column(Enum(SegmentKind, name="segment_kind"))
    heading_level: Mapped[int | None] = mapped_column(Integer, default=None)
    source_text: Mapped[str] = mapped_column(Text, default="")
    translated_text: Mapped[str | None] = mapped_column(Text, default=None)
    # Числовая валидация (roadmap § 3.4 п.3): не сошлось после ре-перевода
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    # bbox, table_rows / table_rows_ru, location (OOXML) и т.п.
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    document: Mapped[Document] = relationship(back_populates="segments")


class Folder(Base):
    """Папки библиотеки (минимум этапа 3; шаринг/права — этап 5)."""

    __tablename__ = "folders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    """RAG-чанк (roadmap § 5 п.1): раздел/таблица со структурными метаданными.

    Двуязычный индекс (§ 5 п.3): text_en/text_ru + два эмбеддинга.
    tsvector-колонка (BM25-контур) создаётся в миграции как generated column —
    в модели не маппится, поиск по ней через raw SQL.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), default="section")  # section | table
    heading_path: Mapped[str] = mapped_column(Text, default="")  # «4 → 4.3 → Таблица 2»
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    page_end: Mapped[int | None] = mapped_column(Integer, default=None)
    text_en: Mapped[str] = mapped_column(Text, default="")
    text_ru: Mapped[str] = mapped_column(Text, default="")
    emb_en: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
    emb_ru: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
    # segment_ids, bbox по страницам — для подсветки цитат в оригинале
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class PageEmbedding(Base):
    """Эмбеддинг страницы-изображения (§ 12.1 шаг 4): визуальный поиск по
    сканам — печати, штампы, чертежи, где OCR теряет."""

    __tablename__ = "page_embeddings"
    __table_args__ = (UniqueConstraint("document_id", "page_idx", name="uq_page_embeddings_doc_page"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_idx: Mapped[int] = mapped_column(Integer)
    emb: Mapped[list[float] | None] = mapped_column(Vector(4096), default=None)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), default="Новый чат")
    # None — чат по всей библиотеке
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), default=None
    )
    # RBAC + субстрат памяти (Этап 0): владелец сессии (OIDC sub; NULL — dev-период)
    # и папка библиотеки (project scope треда для слоя памяти, §15.0).
    owner_sub: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    folder_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"), default=None
    )
    # Суммаризация старой истории (§ 5 п.5): инкрементальная сводка вытеснённых
    # из окна реплик + сколько самых старых реплик в неё уже свёрнуто.
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    summary_msg_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # [{n, chunk_id, document_id, filename, heading_path, page_idx, bbox}]
    citations: Mapped[list[Any] | None] = mapped_column(JSONB, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """Append-only аудит (roadmap § 9): кто загрузил/перевёл/экспортировал/спросил."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_sub: Mapped[str] = mapped_column(String(64))
    username: Mapped[str | None] = mapped_column(String(128), default=None)
    action: Mapped[str] = mapped_column(String(64))  # upload | download | chat_query | …
    object_type: Mapped[str | None] = mapped_column(String(32), default=None)
    object_id: Mapped[str | None] = mapped_column(String(64), default=None)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)


class GlossaryTerm(Base):
    """Утверждённая терминология EN→RU (roadmap § 3.4 п.1)."""

    __tablename__ = "glossary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    en_term: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    ru_term: Mapped[str] = mapped_column(String(256))
    domain: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --- Слой памяти (docs/MEMORY_rev4_mem0_articles.md §3, §15) ------------------
# user_id — text (наш owner_sub / OIDC sub), tenant_id — uuid-константа (single-org);
# embedding — vector(1024) Qwen3-Embedding-0.6B; tsv — generated (в модель не маппится).
MEMORY_DIM = 1024


class MemoryEvent(Base):
    """Сырой эпизод (ground-truth): реплика, загрузка документа, клик по цитате,
    правка. Из событий пересобираются memory_items (§3.2)."""

    __tablename__ = "memory_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str] = mapped_column(String(64))
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    thread_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    event_type: Mapped[str] = mapped_column(Text)
    role: Mapped[str | None] = mapped_column(Text, default=None)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class MemoryItem(Base):
    """Извлечённая память (пересобираема): факт/предпочтение/правило/глоссарий/
    сводка с lifecycle и temporal-валидностью (§3.3)."""

    __tablename__ = "memory_items"
    __table_args__ = (
        CheckConstraint("scope IN ('user','project','document','thread','org')", name="chk_scope"),
        CheckConstraint(
            "kind IN ('preference','fact','glossary','rule','task','correction','summary')",
            name="chk_kind",
        ),
        CheckConstraint("sensitivity IN ('normal','sensitive','secret')", name="chk_sensitivity"),
        CheckConstraint("status IN ('active','superseded','deleted')", name="chk_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str] = mapped_column(String(64))
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    thread_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    scope: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    source_document_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=None
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    sensitivity: Mapped[str] = mapped_column(Text, default="normal")
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    supersedes: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("memory_items.id"), default=None
    )
    status: Mapped[str] = mapped_column(Text, default="active")
    fingerprint: Mapped[str | None] = mapped_column(Text, default=None)
    memory_provider: Mapped[str] = mapped_column(Text, default="internal")
    external_memory_id: Mapped[str | None] = mapped_column(Text, default=None)
    provider_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(MEMORY_DIM), default=None)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class MemoryItemSource(Base):
    """Lineage item↔event (источник истины для purge/пересборки, §3.5)."""

    __tablename__ = "memory_item_sources"

    item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("memory_items.id", ondelete="CASCADE"), primary_key=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("memory_events.id", ondelete="CASCADE"), primary_key=True
    )


class MemoryCandidate(Base):
    """Кандидат до принятия (§3.4): автоэкстрактор пишет сюда, в memory_items
    попадает только после accept/auto_accept через consolidation."""

    __tablename__ = "memory_candidates"
    __table_args__ = (
        CheckConstraint("action IN ('create','update','delete','supersede')", name="chk_cand_action"),
        CheckConstraint(
            "status IN ('pending','accepted','rejected','auto_accepted')", name="chk_cand_status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str] = mapped_column(String(64))
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    thread_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    action: Mapped[str] = mapped_column(Text)
    target_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("memory_items.id"), default=None
    )
    proposed: Mapped[dict[str, Any]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text, default=None)
    fingerprint: Mapped[str | None] = mapped_column(Text, default=None)
    memory_provider: Mapped[str] = mapped_column(Text, default="internal")
    external_memory_id: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(Text, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    decided_by: Mapped[str | None] = mapped_column(Text, default=None)


class MemoryAuditLog(Base):
    """Журнал изменений памяти + gate-блоков + purge (§3.6)."""

    __tablename__ = "memory_audit_log"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str | None] = mapped_column(String(64), default=None)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), default=None)
    action: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(Text)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
