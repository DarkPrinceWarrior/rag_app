"""Схема БД (этап 2): documents + segments + glossary.

Дальше по roadmap добавятся: users, versions, chat_sessions,
chunks(+vector), audit_log. Схемой управляет alembic
(`uv run alembic upgrade head`), create_all больше не используется.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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


class GlossaryTerm(Base):
    """Утверждённая терминология EN→RU (roadmap § 3.4 п.1)."""

    __tablename__ = "glossary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    en_term: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    ru_term: Mapped[str] = mapped_column(String(256))
    domain: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
