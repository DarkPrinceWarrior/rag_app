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
from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
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

    s3_key_original: Mapped[str] = mapped_column(String(1024))
    s3_key_content_list: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_export_docx: Mapped[str | None] = mapped_column(String(1024), default=None)
    # BabelDOC: PDF с сохранённой вёрсткой (mono — только перевод, dual — EN+RU)
    s3_key_export_pdf: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_export_pdf_dual: Mapped[str | None] = mapped_column(String(1024), default=None)
    # OOXML-ветка: переведённый файл исходного формата (docx/xlsx/pptx)
    s3_key_export_source: Mapped[str | None] = mapped_column(String(1024), default=None)

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


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), default="Новый чат")
    # None — чат по всей библиотеке
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), default=None
    )
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
